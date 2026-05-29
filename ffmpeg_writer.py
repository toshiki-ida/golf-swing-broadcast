"""
ffmpeg パイプによる高品質動画エンコーダ

cv2.VideoWriter の代替。H.264 CRF モードで視覚的にほぼ無劣化の出力を行う。

使い方:
    writer = FFmpegWriter("out.mp4", 1920, 1080, 29.97)
    writer.write(frame)   # BGR numpy array
    writer.release()
"""

import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger("ffmpeg")

# ffmpeg 実行パスのキャッシュ
_ffmpeg_path: str | None = None
# ハードウェアエンコーダ検出キャッシュ (None=未検出, ""=なし, "h264_nvenc"等)
_hw_encoder: str | None = None


def find_ffmpeg() -> str | None:
    """ffmpeg の実行パスを検索 (見つからなければ None)

    検索順:
      1. アプリケーション同梱 (PyInstaller onedir — EXEと同じフォルダ)
      2. スクリプトと同じディレクトリ (開発時)
      3. PATH
      4. Windows でよくある場所
    """
    global _ffmpeg_path
    if _ffmpeg_path is not None:
        return _ffmpeg_path if _ffmpeg_path else None

    # 1. PyInstaller _internal ディレクトリ (onedir 同梱)
    if getattr(sys, 'frozen', False):
        internal_dir = Path(sys.executable).parent / "_internal"
        candidate = internal_dir / "ffmpeg.exe"
        if candidate.exists():
            _ffmpeg_path = str(candidate)
            return _ffmpeg_path

    # 2. EXE と同じディレクトリ
    exe_dir = Path(sys.executable).parent
    candidate = exe_dir / "ffmpeg.exe"
    if candidate.exists():
        _ffmpeg_path = str(candidate)
        return _ffmpeg_path

    # 3. スクリプトと同じディレクトリ (開発時)
    script_dir = Path(__file__).parent
    candidate = script_dir / "ffmpeg.exe"
    if candidate.exists():
        _ffmpeg_path = str(candidate)
        return _ffmpeg_path

    # 3. PATH 上を検索
    path = shutil.which("ffmpeg")
    if path:
        _ffmpeg_path = path
        return path

    # 4. Windows でよくある場所
    for p in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]:
        if Path(p).exists():
            _ffmpeg_path = p
            return p

    _ffmpeg_path = ""  # 見つからなかった (空文字でキャッシュ)
    return None


def detect_hw_encoder() -> str:
    """利用可能なH.264ハードウェアエンコーダを検出

    Returns: エンコーダ名 ("h264_nvenc" 等) or "" (なし)
    """
    global _hw_encoder
    if _hw_encoder is not None:
        return _hw_encoder

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        _hw_encoder = ""
        return ""

    # 優先順: NVENC > QSV > AMF
    for enc in ["h264_nvenc", "h264_qsv", "h264_amf"]:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-f", "lavfi", "-i",
                 "nullsrc=s=256x256:d=0.1", "-c:v", enc, "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                _hw_encoder = enc
                log.info(f"HWエンコーダ検出: {enc}")
                return enc
        except Exception:
            continue

    _hw_encoder = ""
    log.info("HWエンコーダなし、libx264 を使用")
    return ""


def _build_encoder_args(crf, preset, hw_encode=False):
    """エンコーダ引数を構築

    hw_encode=True かつ HWエンコーダが利用可能なら NVENC 等を使用。
    NVENCは CRF 非対応のため CQ (Constant Quality) モードで変換。
    preset: NVENC使用時は p1(最速)〜p7(最高品質) にマッピング。
            'ultrafast'→p1, 'fast'→p4, 'medium'→p5, 'slow'→p7
    """
    _nvenc_preset_map = {
        "ultrafast": "p1", "superfast": "p1", "veryfast": "p2",
        "faster": "p3", "fast": "p4", "medium": "p5",
        "slow": "p6", "slower": "p7", "veryslow": "p7",
    }
    if hw_encode:
        enc = detect_hw_encoder()
        if enc == "h264_nvenc":
            nvp = _nvenc_preset_map.get(preset, "p4")
            return ["-c:v", enc, "-preset", nvp, "-rc", "constqp",
                    "-qp", str(crf), "-pix_fmt", "yuv420p"]
        elif enc == "h264_qsv":
            return ["-c:v", enc, "-preset", "fast",
                    "-global_quality", str(crf), "-pix_fmt", "yuv420p"]
        elif enc == "h264_amf":
            return ["-c:v", enc, "-quality", "balanced",
                    "-qp_i", str(crf), "-qp_p", str(crf),
                    "-pix_fmt", "yuv420p"]
    # ソフトウェアフォールバック
    return ["-c:v", "libx264", "-preset", preset,
            "-crf", str(crf), "-pix_fmt", "yuv420p"]


class FFmpegWriter:
    """ffmpeg パイプによる高品質動画ライター

    cv2.VideoWriter と同じインターフェース (write / release / isOpened)。
    H.264 (libx264) CRF エンコードを使用。

    Parameters
    ----------
    output_path : str
        出力ファイルパス (.mp4)
    width, height : int
        フレーム解像度
    fps : float
        フレームレート
    crf : int
        品質 (0=ロスレス, 18=視覚無劣化, 23=デフォルト, 28=低品質)
    preset : str
        エンコード速度 ('ultrafast'〜'veryslow')
        録画用は 'ultrafast', オフライン書き出しは 'medium' 推奨
    """

    def __init__(self, output_path, width, height, fps,
                 crf=18, preset="ultrafast", hw_encode=False,
                 input_yuv=False):
        self._path = str(output_path)
        self._width = width
        self._height = height
        self._opened = False
        self._input_yuv = input_yuv

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise FileNotFoundError(
                "ffmpeg が見つかりません。C:\\ffmpeg\\bin に配置するか PATH に追加してください")

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        enc_args = _build_encoder_args(crf, preset, hw_encode=hw_encode)
        # input_yuv=True: BGR→YUV420P変換してからパイプ転送 (データ量半減: 6MB→3MB/frame)
        in_pix_fmt = "yuv420p" if input_yuv else "bgr24"
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", in_pix_fmt,
            "-r", str(fps),
            "-i", "-",
        ] + enc_args + [self._path]
        log.debug(f"FFmpegWriter: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._opened = True
        # stderr を読み捨てるスレッド (パイプバッファ溢れによるデッドロック防止)
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="ffmpeg-stderr")
        self._stderr_thread.start()

    def _drain_stderr(self):
        """ffmpeg の stderr を継続的に読み取る (バッファ溢れ防止)"""
        try:
            for line in self._proc.stderr:
                if isinstance(line, bytes):
                    line = line.decode(errors="replace")
                self._stderr_lines.append(line.rstrip())
                # 直近500行だけ保持
                if len(self._stderr_lines) > 500:
                    self._stderr_lines = self._stderr_lines[-200:]
        except (ValueError, OSError):
            pass

    def isOpened(self):
        return self._opened and self._proc.poll() is None

    def write(self, frame):
        """BGR フレーム (numpy ndarray) を書き込み"""
        if not self._opened:
            return
        try:
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                import cv2
                frame = cv2.resize(frame, (self._width, self._height))
            if self._input_yuv:
                import cv2
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                self._proc.stdin.write(yuv.tobytes())
            else:
                self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            log.error(f"FFmpegWriter.write 失敗: {e}")
            self._opened = False

    def release(self):
        """書き込みを終了しファイルを確定"""
        if not self._opened:
            return
        self._opened = False
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("FFmpegWriter: タイムアウト、プロセスを強制終了")
            self._proc.kill()
        # stderr スレッドの終了を待機
        if hasattr(self, '_stderr_thread'):
            self._stderr_thread.join(timeout=2.0)
        if self._proc.returncode and self._proc.returncode != 0:
            stderr = "\n".join(self._stderr_lines[-20:]) if self._stderr_lines else ""
            log.error(f"FFmpegWriter 終了コード {self._proc.returncode}: {stderr}")


def ffmpeg_extract(input_path, output_path, in_frame, out_frame, fps,
                   crf=18, preset="medium", hw_encode=False):
    """ffmpeg でフレーム範囲を抽出 (H.264, フレーム精度)

    Parameters
    ----------
    input_path : str  入力動画パス
    output_path : str  出力動画パス
    in_frame, out_frame : int  フレーム範囲 (inclusive)
    fps : float  フレームレート
    crf : int  品質
    preset : str  エンコード速度
    hw_encode : bool  HWエンコーダを使用するか

    Returns
    -------
    bool  成功時 True
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    start_sec = in_frame / fps
    duration_sec = (out_frame - in_frame + 1) / fps

    enc_args = _build_encoder_args(crf, preset, hw_encode=hw_encode)
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_sec:.6f}",       # -i の前 → 入力シーク (高速)
        "-i", str(input_path),
        "-t", f"{duration_sec:.6f}",
    ] + enc_args + [str(output_path)]
    log.debug(f"ffmpeg_extract: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        log.error(f"ffmpeg_extract 失敗: {stderr}")
    return result.returncode == 0
