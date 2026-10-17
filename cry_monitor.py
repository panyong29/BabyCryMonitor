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

APP_VERSION = "1.0.3"

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
    # RTSP 传输协议：udp（局域网推荐，延迟低、丢包不卡死）或 tcp
    "rtsp_transport": "udp",
    # 音频地址：留空 = 与视频共用一路 RTSP（单会话，不额外占带宽）；
    # 摄像头有纯音频子码流时可单独填（如小蚁刷机版的 ch0_1），此时走双会话但音频路只传音频
    "rtsp_audio_url": "",
    "energy_threshold": 0.02,
    "loud_seconds": 2.0,
    "yamnet_threshold": 0.5,
    "cooldown_seconds": 60,
    "volume": 80,
    "muted": False,
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
        # 单会话模式：视频 worker 通过本地回环 TCP 把音频 PCM 交给音频线程，
        # 音频线程不再拉第二路 RTSP，避免重复传输视频浪费带宽
        self._audio_sock_q = queue.Queue()
        self._shared_mode = True
        self.threads = []
        # ffmpeg 子进程登记：退出时统一杀掉，防止孤儿进程占住 RTSP 连接
        self.procs = []
        self.procs_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_alert = 0.0
        self.last_audio_time = 0.0   # 主路最近一次收到音频块的时间
        self.last_audio_peak = 0.0   # 主路最近一块音频的峰值
        self.video_label_img = None
        self._last_frame = None  # 上一帧 PIL 图，供窗口缩放时立即重绘
        self._resize_after_id = None  # 缩放防抖计时器
        # 视频显示尺寸（跟随窗口变化，初始为配置分辨率）
        self._disp_w = self.cfg.get("video_width", 640)
        self._disp_h = self.cfg.get("video_height", 360)
        # 音量（0~100）与静音
        self.volume_var = tk.IntVar(value=int(self.cfg.get("volume", 80)))
        self.mute_var = tk.BooleanVar(value=bool(self.cfg.get("muted", False)))

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

        # 视频区：外层 Frame 负责布局（pack_propagate 关掉，内部图片尺寸
        # 不影响外部布局，打破"图片变大→label 变大→重绘更大图片"的几何反馈回路）
        self.video_frame = tk.Frame(self.root, background="black")
        self.video_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self.video_frame.pack_propagate(False)
        self.video_label = ttk.Label(self.video_frame, text="视频未启动", anchor="center",
                                     background="black", foreground="gray")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_frame.bind("<Configure>", self._on_video_resize)

        info = ttk.Frame(self.root, padding=(8, 0))
        info.pack(fill="x")
        self.mode_label = ttk.Label(info, text="检测模式：--")
        self.mode_label.pack(side="left", padx=4)
        self.audio_label = ttk.Label(info, text="音频流：--")
        self.audio_label.pack(side="left", padx=12)
        self.last_label = ttk.Label(info, text="上次提醒：--")
        self.last_label.pack(side="left", padx=12)
        ttk.Label(info, text="音量").pack(side="left", padx=(12, 2))
        self.volume_scale = ttk.Scale(info, from_=0, to=100, orient="horizontal",
                                      variable=self.volume_var, length=110)
        self.volume_scale.pack(side="left")
        ttk.Checkbutton(info, text="静音", variable=self.mute_var).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w",
                  padding=6).pack(fill="x", side="bottom")

    def _on_video_resize(self, event):
        if event.width > 10 and event.height > 10:
            self._disp_w, self._disp_h = event.width, event.height
            # 防抖：动画/拖拽中频繁触发时，只在停稳 50ms 后重绘一次
            if self._resize_after_id is not None:
                try:
                    self.root.after_cancel(self._resize_after_id)
                except Exception:  # noqa: BLE001
                    pass
            self._resize_after_id = self.root.after(50, self._redraw_video)

    def _redraw_video(self):
        """主线程：把上一帧按显示尺寸等比缩放后显示。"""
        img = self._last_frame
        if img is None:
            return
        from PIL import Image, ImageTk
        lw, lh = self._disp_w, self._disp_h
        if lw > 10 and lh > 10:
            scale = min(lw / img.width, lh / img.height)
            nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            if (nw, nh) != img.size:
                img = img.resize((nw, nh), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self.video_label_img = photo
        self.video_label.config(image=photo, text="")

    # ---------- 子进程管理（防孤儿进程占住 RTSP） ----------
    def _register_proc(self, p):
        with self.procs_lock:
            self.procs.append(p)
        return p

    def _close_proc(self, p):
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    p.kill()
        except Exception:  # noqa: BLE001
            pass
        finally:
            with self.procs_lock:
                if p in self.procs:
                    self.procs.remove(p)

    def _kill_all_procs(self):
        with self.procs_lock:
            procs = list(self.procs)
            self.procs.clear()
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:  # noqa: BLE001
                pass
        for p in procs:
            try:
                p.wait(timeout=3)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass

    def _save_config(self):
        try:
            self.cfg["volume"] = int(self.volume_var.get())
            self.cfg["muted"] = bool(self.mute_var.get())
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass

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
            self._last_frame = self.frame_q.get_nowait()
            self._redraw_video()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ---------- 控制 ----------
    def start(self):
        if self.running or not self.ffmpeg:
            return
        video_url = (self.cfg.get("rtsp_video_url") or "").strip()
        audio_url = (self.cfg.get("rtsp_audio_url") or "").strip()
        # 单会话模式：音频地址为空或与视频地址相同 → 只拉一路 RTSP，
        # 音频从视频流里分流出来，不重复传输视频、不多占带宽
        self._shared_mode = (not audio_url) or (audio_url == video_url)
        logging.info("启动模式: %s",
                     "单会话（音视频共用一路 RTSP）" if self._shared_mode
                     else "双会话（音频走独立子码流）")
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
        self._kill_all_procs()
        for t in self.threads:
            t.join(timeout=3)
        self.threads = []
        # 关闭共享音频 socket 残留
        while True:
            try:
                s = self._audio_sock_q.get_nowait()
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
            except queue.Empty:
                break
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.video_label.config(image="", text="视频未启动")
        self.video_label_img = None
        self._last_frame = None
        self.audio_label.config(text="音频流：--")
        self.set_status("已停止")

    def on_close(self):
        self._save_config()
        self.stop()
        self.root.destroy()

    def _accept_audio_sock(self, srv):
        """后台线程：等 ffmpeg 的音频回环 TCP 连回来，交给音频线程。
        流无音频轨时 accept 超时也不影响视频（视频早就开始播了）。"""
        import socket
        try:
            a_sock, _ = srv.accept()
            a_sock.settimeout(10)
            # 丢掉上一轮重连残留的旧 socket，只保留最新
            while True:
                try:
                    old = self._audio_sock_q.get_nowait()
                    try:
                        old.close()
                    except Exception:  # noqa: BLE001
                        pass
                except queue.Empty:
                    break
            self._audio_sock_q.put(a_sock)
        except socket.timeout:
            logging.warning("音频回环 15 秒未连上（该流可能没有音频轨），本次只传视频")
        except Exception as e:  # noqa: BLE001
            logging.warning("音频回环异常: %s", e)
        finally:
            try:
                srv.close()
            except Exception:  # noqa: BLE001
                pass

    # ---------- 视频 ----------
    def _video_loop(self):
        import socket
        from PIL import Image
        from detection import SAMPLE_RATE
        w, h, fps = (self.cfg["video_width"], self.cfg["video_height"], self.cfg["video_fps"])
        video_url = (self.cfg.get("rtsp_video_url") or "").strip()
        if not video_url:
            self.set_status("视频流地址未配置，请检查 config.json", error=True)
            return
        frame_bytes = w * h * 3
        while self.running and not self.stop_event.is_set():
            srv = None
            try:
                if self._shared_mode:
                    # 单会话：一个 ffmpeg 拉一次流，视频走 stdout 管道，
                    # 音频走本地回环 TCP 分流给音频线程（Windows 下无法双 stdout）
                    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind(("127.0.0.1", 0))
                    srv.listen(1)
                    srv.settimeout(15)
                    a_port = srv.getsockname()[1]
                    cmd = [self.ffmpeg, "-rtsp_transport", self.cfg.get("rtsp_transport", "udp"), "-timeout", "10000000",
                           "-i", video_url,
                           "-map", "0:v", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-an", "-",
                           "-map", "0:a?", "-vn", "-f", "s16le", "-ac", "1",
                           "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
                           f"tcp://127.0.0.1:{a_port}"]
                else:
                    cmd = [self.ffmpeg, "-rtsp_transport", self.cfg.get("rtsp_transport", "udp"), "-timeout", "10000000",
                           "-i", video_url,
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-an", "-"]
                p = self._register_proc(subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=_NO_WINDOW))
                if self._shared_mode:
                    # 音频回环 accept 放后台线程：流无音频轨时不阻塞视频启动
                    threading.Thread(target=self._accept_audio_sock, args=(srv,),
                                     daemon=True).start()
                    srv = None  # 移交后不再由本线程关闭
                self.set_status("视频流已连接" + ("（音视频共用一路）" if self._shared_mode else ""))
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
                self._close_proc(p)
            except Exception as e:  # noqa: BLE001
                logging.warning("视频流异常: %s", e)
            finally:
                if srv is not None:
                    try:
                        srv.close()
                    except Exception:  # noqa: BLE001
                        pass
            if self.running and not self.stop_event.is_set():
                self.set_status("视频流断开，5 秒后重连…（检查 config.json 里 rtsp_video_url 对不对）", error=True)
                self.stop_event.wait(5)

    # ---------- 音频检测 + 实时播放 ----------
    def _audio_loop(self):
        import numpy as np
        from detection import CryDetector, SAMPLE_RATE, CHUNK_SECONDS
        detector = CryDetector(self.cfg)
        self.mode_label.config(text="检测模式：" + detector.mode)
        # 实时播放：有声卡就播，无声卡只做检测不报错
        player = None
        try:
            import sounddevice as sd
            player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="float32", blocksize=int(SAMPLE_RATE * CHUNK_SECONDS))
            player.start()
        except Exception as e:  # noqa: BLE001
            logging.warning("音频播放不可用（只做检测）: %s", e)
            player = None
        chunk_bytes = int(SAMPLE_RATE * CHUNK_SECONDS) * 2

        def handle_pcm(raw):
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            self.last_audio_time = time.time()
            self.last_audio_peak = float(np.max(np.abs(pcm)))
            if player is not None:
                try:
                    gain = 0.0 if self.mute_var.get() else self.volume_var.get() / 100.0
                    player.write((pcm * gain).astype(np.float32))
                except Exception:  # noqa: BLE001
                    pass
            triggered, info = detector.feed(pcm)
            if triggered:
                now = time.time()
                if now - self.last_alert >= self.cfg["cooldown_seconds"]:
                    self.last_alert = now
                    logging.info("哭声触发: %s", info)
                    self.alert_q.put(("🍼 宝宝哭声", "检测到哭声（%s），快去看一下" % info))

        try:
            if self._shared_mode:
                self._audio_loop_shared(chunk_bytes, handle_pcm)
            else:
                self._audio_loop_split(chunk_bytes, handle_pcm)
        finally:
            if player is not None:
                try:
                    player.stop()
                    player.close()
                except Exception:  # noqa: BLE001
                    pass

    def _audio_loop_shared(self, chunk_bytes, handle_pcm):
        """单会话模式：从视频 worker 的本地回环 TCP 读 PCM，不再拉第二路 RTSP。"""
        import socket
        self.audio_label.config(text="音频流：等待视频流…")
        while self.running and not self.stop_event.is_set():
            try:
                sock = self._audio_sock_q.get(timeout=1)
            except queue.Empty:
                continue
            self.audio_label.config(text="音频流：已连接（与视频共用一路）")
            self.set_status("监听中…")
            buf = b""
            try:
                while self.running and not self.stop_event.is_set():
                    try:
                        data = sock.recv(65536)
                    except socket.timeout:
                        continue
                    if not data:
                        break  # 视频端重连或退出，等下一个 socket
                    buf += data
                    while len(buf) >= chunk_bytes:
                        handle_pcm(buf[:chunk_bytes])
                        buf = buf[chunk_bytes:]
            except Exception as e:  # noqa: BLE001
                logging.warning("共享音频流异常: %s", e)
            finally:
                try:
                    sock.close()
                except Exception:  # noqa: BLE001
                    pass
            if self.running and not self.stop_event.is_set():
                self.audio_label.config(text="音频流：等待视频流重连…")

    def _audio_loop_split(self, chunk_bytes, handle_pcm):
        """双会话模式：音频地址是独立子码流（纯音频）时，自己拉一路 RTSP。
        该路只传输音频，无视频带宽浪费。"""
        from detection import SAMPLE_RATE
        audio_url = (self.cfg.get("rtsp_audio_url") or "").strip()
        if not audio_url:
            self.audio_label.config(text="音频流：未配置")
            self.set_status("音频流地址未配置，请检查 config.json", error=True)
            return
        while self.running and not self.stop_event.is_set():
            try:
                cmd = [self.ffmpeg, "-rtsp_transport", self.cfg.get("rtsp_transport", "udp"), "-timeout", "10000000",
                       "-i", audio_url,
                       "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                       "-acodec", "pcm_s16le", "-"]
                p = self._register_proc(subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=_NO_WINDOW))
                self.audio_label.config(text="音频流：已连接")
                self.set_status("监听中…")
                while self.running and not self.stop_event.is_set():
                    raw = p.stdout.read(chunk_bytes)
                    if len(raw) < chunk_bytes:
                        break
                    handle_pcm(raw)
                self._close_proc(p)
            except Exception as e:  # noqa: BLE001
                logging.warning("音频流异常: %s", e)
            if self.running and not self.stop_event.is_set():
                self.audio_label.config(text="音频流：重连中…")
                self.stop_event.wait(5)

    # ---------- 自检 ----------
    def selftest_audio(self):
        # 监控运行时主路已有音频数据：直接用实时数据判断，不再开第二路 RTSP
        # （yi-hack 的 RTSP 服务弱，第二路会被饿死导致误报无音频）
        if self.running and (time.time() - self.last_audio_time) < 5:
            peak = self.last_audio_peak
            self.set_status("音频流正常（主路实时，峰值 %.3f）" % peak)
            messagebox.showinfo("自检", "音频流正常！\n\n主路实时音频峰值 %.3f。\n在摄像头旁拍手，峰值会明显跳动。" % peak)
            return
        def _run():
            audio_url = (self.cfg.get("rtsp_audio_url") or "").strip() \
                or (self.cfg.get("rtsp_video_url") or "").strip()
            if not audio_url:
                messagebox.showinfo("自检", "未配置音频流地址，请先在 config.json 里填写 rtsp 地址。")
                return
            self.set_status("正在检测音频流…")
            try:
                cmd = [self.ffmpeg, "-rtsp_transport", self.cfg.get("rtsp_transport", "udp"), "-timeout", "10000000",
                       "-i", audio_url,
                       "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000",
                       "-acodec", "pcm_s16le", "-t", "4", "-"]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_NO_WINDOW)
                try:
                    data, err = p.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    data, err = p.communicate()
                err_txt = (err or b"").decode("utf-8", "ignore")[-800:]
                if len(data) > 16000:
                    import numpy as np
                    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    peak = float(np.max(np.abs(pcm)))
                    self.set_status("音频流正常（4 秒收到 %d 字节，峰值 %.3f）" % (len(data), peak))
                    messagebox.showinfo("自检", "音频流正常！\n\n4 秒收到 %d 字节，峰值 %.3f。\n在摄像头旁拍手，峰值会明显跳动。" % (len(data), peak))
                else:
                    self.set_status("音频流无数据", error=True)
                    messagebox.showwarning("自检", "连上了但没收到音频数据（%d 字节）。\n\nffmpeg 输出：\n%s" % (len(data), err_txt or "（无）"))
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
