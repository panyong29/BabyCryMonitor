"""宝宝哭声监控 - Windows 客户端 v1.0

拉取摄像头 RTSP 流：视频窗口实时看，纯音频流做哭声检测，
检测到哭声后 Windows 右下角弹窗提醒。

检测：能量门 -> YAMNet AI 确认（缺模型时自动降级为纯音量模式）。
"""
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

APP_VERSION = "1.0.1"

# Windows 下拉起 ffmpeg（控制台程序）时不弹黑框；非 Windows 为 0 无影响
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
APP_NAME = "宝宝哭声监控"

# ---------- 路径 ----------
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(BASE_DIR, "baby_cry_monitor.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)

DEFAULT_CONFIG = {
    "rtsp_video_url": "rtsp://192.168.1.100:554/stream1",
    "rtsp_audio_url": "rtsp://192.168.1.100:554/stream1",
    "energy_threshold": 0.02,
    "loud_seconds": 2.0,
    "yamnet_threshold": 0.5,
    "cooldown_seconds": 60,
    "video_width": 640,
    "video_height": 360,
    "video_fps": 8,
}


def load_config():
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            merged = dict(DEFAULT_CONFIG)
            merged.update(cfg)
            return merged
        except Exception as e:  # noqa: BLE001
            logging.warning("配置文件读取失败，用默认配置: %s", e)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    return dict(DEFAULT_CONFIG)


def notify(title, msg):
    """右下角 Toast；失败则退化为置顶弹窗。"""
    def _show():
        try:
            from win11toast import toast
            toast(title, msg)
            return
        except Exception as e:  # noqa: BLE001
            logging.warning("toast 失败: %s", e)
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showwarning(title, msg)
            root.destroy()
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_show, daemon=True).start()


class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("700x560")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.cfg = load_config()
        self.running = False
        self.ffmpeg = None
        try:
            from ffmpeg_bin import find_ffmpeg
            self.ffmpeg = find_ffmpeg()
        except Exception as e:  # noqa: BLE001
            logging.error("ffmpeg 定位失败: %s", e)

        self.alert_q = queue.Queue()
        self.frame_q = queue.Queue(maxsize=2)
        self.status_q = queue.Queue()
        self.threads = []
        self.stop_event = threading.Event()
        self.last_alert = 0.0
        self.video_label_img = None
        # 视频显示尺寸（跟随窗口变化，初始为配置分辨率）
        self._disp_w = self.cfg.get("video_width", 640)
        self._disp_h = self.cfg.get("video_height", 360)

        self._build_ui()
        self._poll()
        if not self.ffmpeg:
            self.set_status("未找到 ffmpeg，监控无法启动", error=True)

    # ---------- 界面 ----------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.btn_start = ttk.Button(top, text="开始监控", command=self.start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(top, text="停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(top, text="测试弹窗", command=lambda: notify("🍼 测试", "弹窗正常")).pack(side="left", padx=4)
        ttk.Button(top, text="检测音频流", command=self.selftest_audio).pack(side="left", padx=4)
        self.autostart_var = tk.BooleanVar(value=self._autostart_enabled())
        ttk.Checkbutton(top, text="开机自启", variable=self.autostart_var,
                        command=self._toggle_autostart).pack(side="right", padx=4)

        self.video_label = ttk.Label(self.root, text="视频未启动", anchor="center",
                                     background="black", foreground="gray",
                                     width=80)
        self.video_label.pack(fill="both", expand=True, padx=8, pady=4)
        self.video_label.bind("<Configure>", self._on_video_resize)

        info = ttk.Frame(self.root, padding=(8, 0))
        info.pack(fill="x")
        self.mode_label = ttk.Label(info, text="检测模式：--")
        self.mode_label.pack(side="left", padx=4)
        self.audio_label = ttk.Label(info, text="音频流：--")
        self.audio_label.pack(side="left", padx=12)
        self.last_label = ttk.Label(info, text="上次提醒：--")
        self.last_label.pack(side="left", padx=12)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w",
                  padding=6).pack(fill="x", side="bottom")

    def _on_video_resize(self, event):
        if event.width > 10 and event.height > 10:
            self._disp_w, self._disp_h = event.width, event.height

    def set_status(self, text, error=False):
        self.status_q.put((text, error))

    def _poll(self):
        try:
            while True:
                text, error = self.status_q.get_nowait()
                self.status_var.set(text)
        except queue.Empty:
            pass
        try:
            while True:
                title, msg = self.alert_q.get_nowait()
                self.last_label.config(text="上次提醒：" + time.strftime("%H:%M:%S"))
                notify(title, msg)
        except queue.Empty:
            pass
        try:
            from PIL import Image, ImageTk
            img = self.frame_q.get_nowait()
            # 按显示区域等比缩放（主线程做，tkinter 非线程安全）
            lw, lh = self._disp_w, self._disp_h
            if lw > 10 and lh > 10 and (img.width, img.height) != (lw, lh):
                scale = min(lw / img.width, lh / img.height)
                nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
                img = img.resize((nw, nh), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self.video_label_img = photo
            self.video_label.config(image=photo, text="")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ---------- 控制 ----------
    def start(self):
        if self.running or not self.ffmpeg:
            return
        self.running = True
        self.stop_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.set_status("正在启动…")
        self.threads = [
            threading.Thread(target=self._video_loop, daemon=True),
            threading.Thread(target=self._audio_loop, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def stop(self):
        self.running = False
        self.stop_event.set()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.video_label.config(image="", text="视频未启动")
        self.video_label_img = None
        self.set_status("已停止")

    def on_close(self):
        self.stop()
        time.sleep(0.3)
        self.root.destroy()

    # ---------- 视频 ----------
    def _video_loop(self):
        from PIL import Image, ImageTk
        w, h, fps = (self.cfg["video_width"], self.cfg["video_height"], self.cfg["video_fps"])
        frame_bytes = w * h * 3
        while self.running and not self.stop_event.is_set():
            try:
                cmd = [self.ffmpeg, "-rtsp_transport", "tcp", "-i", self.cfg["rtsp_video_url"],
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                       "-r", str(fps), "-an", "-"]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=_NO_WINDOW)
                self.set_status("视频流已连接")
                while self.running and not self.stop_event.is_set():
                    raw = p.stdout.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        break
                    img = Image.frombytes("RGB", (w, h), raw, "raw", "BGR")
                    # PIL Image 跨线程传递安全；PhotoImage 必须在主线程创建（tkinter 非线程安全）
                    try:
                        self.frame_q.put_nowait(img)
                    except queue.Full:
                        try:
                            self.frame_q.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_q.put_nowait(img)
                p.terminate()
            except Exception as e:  # noqa: BLE001
                logging.warning("视频流异常: %s", e)
            if self.running:
                self.set_status("视频流断开，5 秒后重连…", error=True)
                time.sleep(5)

    # ---------- 音频检测 ----------
    def _audio_loop(self):
        import numpy as np
        from detection import CryDetector, SAMPLE_RATE, CHUNK_SECONDS
        detector = CryDetector(self.cfg)
        self.mode_label.config(text="检测模式：" + detector.mode)
        chunk_bytes = int(SAMPLE_RATE * CHUNK_SECONDS) * 2
        while self.running and not self.stop_event.is_set():
            try:
                cmd = [self.ffmpeg, "-rtsp_transport", "tcp", "-i", self.cfg["rtsp_audio_url"],
                       "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                       "-acodec", "pcm_s16le", "-"]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=_NO_WINDOW)
                self.audio_label.config(text="音频流：已连接")
                self.set_status("监听中…")
                while self.running and not self.stop_event.is_set():
                    raw = p.stdout.read(chunk_bytes)
                    if len(raw) < chunk_bytes:
                        break
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    triggered, info = detector.feed(pcm)
                    if triggered:
                        now = time.time()
                        if now - self.last_alert >= self.cfg["cooldown_seconds"]:
                            self.last_alert = now
                            logging.info("哭声触发: %s", info)
                            self.alert_q.put(("🍼 宝宝哭声", "检测到哭声（%s），快去看一下" % info))
                p.terminate()
            except Exception as e:  # noqa: BLE001
                logging.warning("音频流异常: %s", e)
            if self.running:
                self.audio_label.config(text="音频流：重连中…")
                time.sleep(5)

    # ---------- 自检 ----------
    def selftest_audio(self):
        def _run():
            self.set_status("正在检测音频流…")
            try:
                cmd = [self.ffmpeg, "-rtsp_transport", "tcp", "-i", self.cfg["rtsp_audio_url"],
                       "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000",
                       "-acodec", "pcm_s16le", "-t", "4", "-"]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=_NO_WINDOW)
                data = p.stdout.read()
                p.wait(timeout=10)
                if len(data) > 16000:
                    import numpy as np
                    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    peak = float(np.max(np.abs(pcm)))
                    self.set_status("音频流正常（4 秒收到 %d 字节，峰值 %.3f）" % (len(data), peak))
                    messagebox.showinfo("自检", "音频流正常！\n\n4 秒收到 %d 字节，峰值 %.3f。\n在摄像头旁拍手，峰值会明显跳动。" % (len(data), peak))
                else:
                    self.set_status("音频流无数据", error=True)
                    messagebox.showwarning("自检", "连上了但没收到音频数据。\n\n检查摄像头后台的音频（伴音）开关是否打开。")
            except Exception as e:  # noqa: BLE001
                self.set_status("音频流连接失败", error=True)
                messagebox.showerror("自检", "音频流连接失败：\n%s\n\n检查 config.json 里的地址和摄像头 IP。" % e)
        threading.Thread(target=_run, daemon=True).start()

    # ---------- 开机自启 ----------
    def _autostart_enabled(self):
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Run")
            winreg.QueryValueEx(key, APP_NAME)
            winreg.CloseKey(key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _toggle_autostart(self):
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Run",
                                 0, winreg.KEY_SET_VALUE)
            if self.autostart_var.get():
                exe = sys.executable if getattr(sys, "frozen", False) else ""
                if exe:
                    winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, exe)
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        except Exception as e:  # noqa: BLE001
            messagebox.showwarning("开机自启", "设置失败：%s" % e)
            self.autostart_var.set(self._autostart_enabled())


def main():
    root = tk.Tk()
    MonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
