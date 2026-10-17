"""两级哭声检测：能量门（常开，便宜）-> YAMNet AI 确认（离线小模型）。

音频约定：16kHz 单声道 float32，取值范围 [-1, 1]。
YAMNet 不可用（缺模型/缺推理库）时自动降级为纯音量模式，绝不崩溃。
"""
import csv
import os
import sys

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
BUFFER_SECONDS = 4
AI_WINDOW_SECONDS = 3

BABY_CRY_LABEL = "Baby cry, infant cry"


def _asset(name):
    """按优先级找打包资源：_MEIPASS > 本文件目录 > 当前工作目录。"""
    rel = os.path.join("assets", name)
    search = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        search.append(os.path.join(meipass, rel))
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
