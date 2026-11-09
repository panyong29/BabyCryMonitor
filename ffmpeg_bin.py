"""定位 ffmpeg 可执行文件。

防火墙弹窗根因：PyInstaller --onefile 每次启动都把 ffmpeg.exe 解到
不同的 %TEMP%\\_MEIxxxxx 临时目录，而 Windows 防火墙规则是按程序路径记的——
上次点的"允许"管的是上次那个临时路径，下次换路径就又弹。
治法（仅打包版）：首次启动把内置 ffmpeg 拷一份到固定路径
%LOCALAPPDATA%\\BabyCryMonitor\\ffmpeg.exe，之后一直用这份；
防火墙规则认固定路径，点一次"允许"就不再弹。

优先级：固定路径副本 > PyInstaller 打包目录 > exe 所在目录 > imageio-ffmpeg 自带 > 系统 PATH。
"""
import os
import shutil
import sys

_STABLE_DIR_NAME = "BabyCryMonitor"


def _stable_ffmpeg_path():
    """固定路径副本的位置；非 Windows 返回 None。"""
    if sys.platform != "win32":
        return None
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return os.path.join(base, _STABLE_DIR_NAME, "ffmpeg.exe")


def _bundled_ffmpeg():
    """不经过固定路径副本，直接找内置/现成的 ffmpeg（旧逻辑）。"""
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


def _ensure_stable_copy(src):
    """把 src 拷到固定路径（先写临时文件再原子改名，避免拷一半的残文件），
    返回固定路径。任何一步失败返回 None（调用方回退到旧逻辑）。"""
    dst = _stable_ffmpeg_path()
    if not dst or not src or not os.path.isfile(src):
        return None
    try:
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return dst  # 已有副本且和来源同一大小，直接用，不重复拷
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return dst
    except Exception:
        try:
            if os.path.isfile(dst + ".tmp"):
                os.remove(dst + ".tmp")
        except Exception:
            pass
        return None


def find_ffmpeg():
    # 打包版 Windows：优先固定路径副本，治防火墙每次都弹
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        stable = _stable_ffmpeg_path()
        if stable and os.path.isfile(stable):
            bundled = _bundled_ffmpeg()
            # 内置版本变了（文件大小变了）→ 重新拷一份；否则直接用旧副本
            if bundled and os.path.getsize(stable) != os.path.getsize(bundled):
                stable = _ensure_stable_copy(bundled) or stable
            return stable
        bundled = _bundled_ffmpeg()
        copied = _ensure_stable_copy(bundled) if bundled else None
        if copied:
            return copied
    return _bundled_ffmpeg()
