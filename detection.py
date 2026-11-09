"""两级哭声检测：能量门（常开，便宜）-> YAMNet AI 确认（离线小模型）。

音频约定：16kHz 单声道 float32，取值范围 [-1, 1]。
YAMNet 不可用（缺模型/缺推理库）时自动降级为纯音量模式，绝不崩溃。
"""
import csv
import os
import sys
import time

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
BUFFER_SECONDS = 4
AI_WINDOW_SECONDS = 3

BABY_CRY_LABEL = "Baby cry, infant cry"


def _asset(name):
    """按优先级找打包资源：_MEIPASS > exe 所在目录（精简版：用户自己下载的
    资源放 exe 旁）> 本文件目录 > 当前工作目录。"""
    rel = os.path.join("assets", name)
    search = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        search.append(os.path.join(meipass, rel))
    if getattr(sys, "frozen", False):
        # 精简版：yamnet.tflite 由使用者自己下载，放 exe 旁的 assets/ 里
        exe_dir = os.path.dirname(sys.executable)
        search.append(os.path.join(exe_dir, rel))
        search.append(os.path.join(exe_dir, name))
    search.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), rel))
    search.append(os.path.join(os.getcwd(), rel))
    for p in search:
        if os.path.isfile(p):
            return p
    return None


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))


class YamnetDetector:
    """YAMNet TFLite：521 类音频分类，取 Baby cry 分数。"""

    def __init__(self):
        self.available = False
        self.error = ""
        try:
            model_path = _asset("yamnet.tflite")
            labels_path = _asset("yamnet_class_map.csv")
            if not model_path or not labels_path:
                self.error = "模型文件缺失"
                return
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tflite_runtime.interpreter import Interpreter  # 旧包名兜底
            self.interp = Interpreter(model_path=model_path)
            in_d = self.interp.get_input_details()[0]
            # 动态输入：固定为 3 秒窗口，避免每次推理重分配
            self.window_n = SAMPLE_RATE * 3
            try:
                self.interp.resize_tensor_input(in_d["index"], [self.window_n], strict=False)
            except Exception:
                pass
            self.interp.allocate_tensors()
            in_d = self.interp.get_input_details()[0]
            out_ds = self.interp.get_output_details()
            self.in_index = in_d["index"]
            # scores 输出：二维、最后一维 521
            self.scores_index = None
            for d in out_ds:
                shape = list(d["shape"])
                if len(shape) == 2 and shape[1] == 521:
                    self.scores_index = d["index"]
                    break
            if self.scores_index is None:
                self.error = "模型输出结构不符"
                return
            self.cry_index = None
            with open(labels_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("display_name") == BABY_CRY_LABEL:
                        self.cry_index = int(row["index"])
                        break
            if self.cry_index is None:
                self.error = "标签表无哭声类"
                return
            self.available = True
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:120]

    def score(self, waveform):
        """输入 16k 单声道 float32（取后 3 秒），返回哭声分数 0~1（多帧取平均）。"""
        if not self.available:
            return 0.0
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        n = getattr(self, "window_n", None) or wav.size
        if wav.size < n:
            wav = np.pad(wav, (n - wav.size, 0))
        else:
            wav = wav[-n:]
        self.interp.set_tensor(self.in_index, wav)
        self.interp.invoke()
        scores = self.interp.get_tensor(self.scores_index)
        return float(np.mean(scores[:, self.cry_index]))


class CryDetector:
    """流式检测器：每 0.5s 喂一块音频，触发时返回 ('crying', 分数)。"""

    def __init__(self, cfg):
        self.sr = SAMPLE_RATE
        self.energy_threshold = float(cfg.get("energy_threshold", 0.02))
        self.loud_seconds = float(cfg.get("loud_seconds", 2.0))
        self.yamnet_threshold = float(cfg.get("yamnet_threshold", 0.5))
        self.chunk_n = int(self.sr * CHUNK_SECONDS)
        self.buf = np.zeros(self.sr * BUFFER_SECONDS, dtype=np.float32)
        self.loud_run = 0.0
        self.yamnet = YamnetDetector()
        self.mode = "AI+音量" if self.yamnet.available else "纯音量"
        if not self.yamnet.available:
            print("[detect] YAMNet 不可用(%s)，降级为纯音量模式" % self.yamnet.error)

    def feed(self, chunk):
        """chunk: 0.5s float32 mono。返回 (triggered: bool, info: str)。"""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.chunk_n:
            return False, ""
        self.buf = np.roll(self.buf, -self.chunk_n)
        self.buf[-self.chunk_n:] = chunk
        r = rms(chunk)
        if r < self.energy_threshold:
            self.loud_run = 0.0
            return False, ""
        self.loud_run += CHUNK_SECONDS
        if self.loud_run < self.loud_seconds:
            return False, ""
        if self.yamnet.available:
            window = self.buf[-self.sr * AI_WINDOW_SECONDS:]
            s = self.yamnet.score(window)
            if s >= self.yamnet_threshold:
                self.loud_run = 0.0
                return True, "AI哭声分数 %.2f" % s
            return False, ""
        self.loud_run = 0.0
        return True, "音量触发 RMS %.3f" % r


def _parse_roi(raw):
    """校验 motion_roi：[x0, y0, x1, y1]，均为 0~1 的相对坐标。
    非法/面积过小返回 None（= 全幅检测）。"""
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except Exception:  # noqa: BLE001
        return None
    x0, y0 = max(0.0, min(x0, 1.0)), max(0.0, min(y0, 1.0))
    x1, y1 = max(0.0, min(x1, 1.0)), max(0.0, min(y1, 1.0))
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return None
    return [x0, y0, x1, y1]


class MotionDetector:
    """帧差法动作检测：只看婴儿床区域内的画面变化。

    输入：每帧一张小灰度图（调用方先把 640x360 帧缩到 160x90 左右，省 CPU）。
    输出事件：None / "motion"（持续动作）/ "vigorous"（剧烈动作）。
    开灯、红外夜视切换这类整幅亮度跳变会直接跳过该帧，不误报。
    注意：它只能报"动了"。吐奶/尿湿是"一直不走的变化"，不是动作，
    归 StainDetector（持续变化监测）管，见下。
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("motion_enabled", True))
        self.roi = _parse_roi(cfg.get("motion_roi"))
        self.threshold = float(cfg.get("motion_threshold", 0.03))
        self.high_threshold = float(cfg.get("motion_high_threshold", 0.12))
        self.pixel_diff = int(cfg.get("motion_pixel_diff", 25))
        self.sustain = float(cfg.get("motion_sustain_seconds", 2.0))
        self.high_sustain = float(cfg.get("motion_high_sustain_seconds", 1.0))
        self.cooldown = float(cfg.get("motion_cooldown_seconds", 60))
        self.brightness_jump = float(cfg.get("motion_brightness_jump", 20))
        self.prev = None
        self._last_t = None
        self.motion_run = 0.0
        self.high_run = 0.0
        self.last_alert = float("-inf")  # 从未报过；不用 0.0，避免 now 很小时被冷却挡住
        self.last_score = 0.0
        self.last_state = "quiet"  # quiet | motion | vigorous | lighting

    def set_roi(self, roi):
        self.roi = _parse_roi(roi)
        self.reset()

    def reset(self):
        self.prev = None
        self._last_t = None
        self.motion_run = 0.0
        self.high_run = 0.0
        self.last_state = "quiet"

    def _region(self, gray):
        """按 ROI 裁一块；无 ROI 返回整幅。"""
        if not self.roi:
            return gray
        h, w = gray.shape[:2]
        x0 = max(0, int(self.roi[0] * w))
        y0 = max(0, int(self.roi[1] * h))
        x1 = max(x0 + 1, int(self.roi[2] * w))
        y1 = max(y0 + 1, int(self.roi[3] * h))
        return gray[y0:y1, x0:x1]

    def feed(self, gray, dt=None, now=None):
        """喂一帧小灰度图（2D uint8 ndarray）。
        返回 (event, score, state)：event 为 None / "motion" / "vigorous"；
        score 是 ROI 内变化像素占比；state 是当前状态。"""
        if not self.enabled:
            return None, 0.0, "quiet"
        gray = np.asarray(gray)
        if gray.ndim != 2 or gray.size == 0:
            return None, 0.0, "quiet"
        if now is None:
            now = time.time()
        if dt is None:
            dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray.copy()
            self.motion_run = 0.0
            self.high_run = 0.0
            return None, 0.0, "quiet"
        # 整幅亮度跳变（开灯、红外夜视切换）：跳过该帧并以当前帧为新基准
        mean_shift = abs(float(gray.mean()) - float(self.prev.mean()))
        if mean_shift > self.brightness_jump:
            self.prev = gray.copy()
            self.motion_run = 0.0
            self.high_run = 0.0
            self.last_state = "lighting"
            return None, 0.0, "lighting"
        region = self._region(gray)
        prev_region = self._region(self.prev)
        diff = np.abs(region.astype(np.int16) - prev_region.astype(np.int16))
        score = float((diff > self.pixel_diff).mean())
        self.prev = gray.copy()
        self.last_score = score
        if score >= self.high_threshold:
            self.high_run += dt
            self.motion_run += dt
        elif score >= self.threshold:
            self.motion_run += dt
            self.high_run = 0.0
        else:
            self.motion_run = 0.0
            self.high_run = 0.0
        event = None
        if self.high_run >= self.high_sustain and now - self.last_alert >= self.cooldown:
            event = "vigorous"
        elif self.motion_run >= self.sustain and now - self.last_alert >= self.cooldown:
            event = "motion"
        if event is not None:
            self.last_alert = now
            self.motion_run = 0.0
            self.high_run = 0.0
        self.last_state = ("vigorous" if self.high_run > 0
                           else "motion" if self.motion_run > 0 else "quiet")
        return event, score, self.last_state


class StainDetector:
    """持续变化监测（吐奶/尿湿试验）：复用婴儿床区域（motion_roi，不再框选），
    白天才测；夜里红外画面是灰度，饱和度低，直接跳过。

    原理：跟几分钟前的参考帧比，区域内新出现且一直不走的亮斑判"疑似吐奶"、
    暗斑判"疑似尿湿"。跟动作检测不同：奶渍/水渍出现后就不动，帧差抓不到，
    只能靠"跟旧参考比 + 持续确认"。
    试验模式（默认开）：只记日志、存截图，不弹窗；跑几天看误报率再决定
    要不要开提醒。注意：宝宝挪位置后身体停在新地方，也会形成一块一直
    不走的亮/暗斑，可能误报——试验期靠截图人工判断。
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("stain_enabled", True))
        self.roi = _parse_roi(cfg.get("motion_roi"))
        self.bright_threshold = int(cfg.get("stain_bright_threshold", 25))
        self.dark_threshold = int(cfg.get("stain_dark_threshold", 25))
        self.min_fraction = float(cfg.get("stain_min_fraction", 0.005))
        self.sustain = float(cfg.get("stain_sustain_seconds", 60.0))
        self.ref_interval = float(cfg.get("stain_ref_interval_seconds", 600.0))
        self.cooldown = float(cfg.get("stain_cooldown_seconds", 600.0))
        self.saturation_threshold = float(cfg.get("stain_saturation_threshold", 8.0))
        self.brightness_jump = float(cfg.get("stain_brightness_jump", 20))
        self.trial_mode = bool(cfg.get("stain_trial_mode", True))
        self.ref = None
        self._ref_time = 0.0
        self._cand_kind = None
        self._cand_start = 0.0
        self.last_alert = float("-inf")  # 从未报过；不用 0.0，避免 now 很小时被冷却挡住
        self.last_score = 0.0
        self.last_state = "idle"  # idle | watching | candidate | lighting

    def set_roi(self, roi):
        self.roi = _parse_roi(roi)
        self.reset()

    def reset(self):
        self.ref = None
        self._ref_time = 0.0
        self._cand_kind = None
        self._cand_start = 0.0
        self.last_state = "idle"

    @staticmethod
    def daytime_score(bgr):
        """色彩分：像素通道间最大差的均值。白天彩色画面高，夜里红外灰度接近 0。
        输入 160x90 左右的小彩色图（RGB 或 BGR 都行，通道顺序不影响最大差）。"""
        bgr = np.asarray(bgr)
        if bgr.ndim != 3 or bgr.shape[2] < 3 or bgr.size == 0:
            return 0.0
        spread = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
        return float(spread.mean())

    def is_day(self, bgr):
        return self.daytime_score(bgr) >= self.saturation_threshold

    def _region(self, gray):
        """按 ROI 裁一块；无 ROI 返回整幅。"""
        if not self.roi:
            return gray
        h, w = gray.shape[:2]
        x0 = max(0, int(self.roi[0] * w))
        y0 = max(0, int(self.roi[1] * h))
        x1 = max(x0 + 1, int(self.roi[2] * w))
        y1 = max(y0 + 1, int(self.roi[3] * h))
        return gray[y0:y1, x0:x1]

    def feed(self, gray, now=None):
        """喂一帧小灰度图（2D uint8 ndarray，白天彩色画面的灰度版）。
        返回 (event, score, state)：event 为 None / "spit_up" / "wet"；
        score 是 ROI 内变化像素占比；state 是当前状态。"""
        if not self.enabled:
            return None, 0.0, "idle"
        gray = np.asarray(gray)
        if gray.ndim != 2 or gray.size == 0:
            return None, 0.0, "idle"
        if now is None:
            now = time.time()
        if self.ref is None or self.ref.shape != gray.shape:
            self.ref = gray.copy()
            self._ref_time = now
            self._cand_kind = None
            self.last_state = "watching"
            return None, 0.0, "watching"
        # 整幅亮度跳变（开灯、拉窗帘）：以当前帧为新参考，不误报
        mean_shift = abs(float(gray.mean()) - float(self.ref.mean()))
        if mean_shift > self.brightness_jump:
            self.ref = gray.copy()
            self._ref_time = now
            self._cand_kind = None
            self.last_state = "lighting"
            return None, 0.0, "lighting"
        # 无可疑变化超过 10 分钟：刷新参考帧，跟上被子、光线这类慢变化
        if self._cand_kind is None and now - self._ref_time > self.ref_interval:
            self.ref = gray.copy()
            self._ref_time = now
            self.last_state = "watching"
            return None, 0.0, "watching"
        region = self._region(gray)
        ref_region = self._region(self.ref)
        diff = region.astype(np.int16) - ref_region.astype(np.int16)
        bright_frac = float((diff > self.bright_threshold).mean())
        dark_frac = float((diff < -self.dark_threshold).mean())
        score = max(bright_frac, dark_frac)
        self.last_score = score
        kind = None
        if bright_frac >= self.min_fraction and bright_frac >= dark_frac:
            kind = "spit_up"
        elif dark_frac >= self.min_fraction:
            kind = "wet"
        if kind is None:
            self._cand_kind = None
            self.last_state = "watching"
            return None, score, "watching"
        if self._cand_kind != kind:
            self._cand_kind = kind
            self._cand_start = now
            self.last_state = "candidate"
            return None, score, "candidate"
        self.last_state = "candidate"
        if now - self._cand_start < self.sustain:
            return None, score, "candidate"
        if now - self.last_alert < self.cooldown:
            return None, score, "candidate"
        # 持续够久：报一次，并把当前画面吸进参考帧，避免同一块渍反复报
        self.last_alert = now
        self.ref = gray.copy()
        self._ref_time = now
        self._cand_kind = None
        self.last_state = "watching"
        return kind, score, kind
