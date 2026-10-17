"""定位 ffmpeg 可执行文件。

优先级：PyInstaller 打包目录 > exe 所在目录 > imageio-ffmpeg 自带 > 系统 PATH。
"""
import os
import shutil
import sys


def find_ffmpeg():
    candidates = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidates.append(os.path.join(bundle, "ffmpeg.exe"))
        candidates.append(os.path.join(bundle, "ffmpeg"))
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(exe_dir, "ffmpeg.exe"))
    candidates.append(os.path.join(exe_dir, "ffmpeg"))
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None
