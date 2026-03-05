"""
録画モジュール

DeckLinkまたはフォールバック入力からの映像をファイルに記録する。
グローウィング対応: 録画中のフレームをJPEGバッファに保持し、
録画しながらスクラブ/In/Out/クリップ切り出しが可能。
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np


# プレビュー用縮小解像度
_PREVIEW_W = 480
_PREVIEW_H = 270

# JPEG圧縮品質 (フルレスバッファ用)
_JPEG_QUALITY = 85


class Recorder:
    """入力映像のRECORD/STOP制御 + グローウィングバッファ"""

    MAX_BUFFER_FRAMES = 1800  # バッファ上限 (~60秒 @30fps)

    def __init__(self, output_dir: str, width=1920, height=1080, fps=29.97):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps

        self._recording = False
        self._writer = None
        self._current_path = None
        self._frame_count = 0
        self._start_time = None
        self._lock = threading.Lock()

        # グローウィングバッファ
        self._preview_buffer = []   # 縮小フレーム (numpy, 480x270) - スクラブ用
        self._fullres_buffer = []   # JPEG bytes - クリップ切り出し用
        # In/Out (グローウィング中に設定可能)
        self._growing_in = 0
        self._growing_out = -1

    @property
    def is_recording(self):
        return self._recording

    @property
    def current_file(self):
        return self._current_path

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def duration_sec(self):
        if self._start_time and self._recording:
            return time.time() - self._start_time
        return self._frame_count / max(self.fps, 1)

    @property
    def buffered_frame_count(self):
        return len(self._preview_buffer)

    @property
    def growing_in(self):
        return self._growing_in

    @growing_in.setter
    def growing_in(self, v):
        self._growing_in = max(0, v)

    @property
    def growing_out(self):
        return self._growing_out

    @growing_out.setter
    def growing_out(self, v):
        self._growing_out = v

    def start_recording(self, filename=None, fps=None):
        """録画開始

        fps: 指定時はこのFPSで録画 (インタレース入力 59.94fps 等)
        """
        with self._lock:
            if self._recording:
                return None

            if fps is not None:
                self.fps = fps

            if filename is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"rec_{ts}.mp4"

            self._current_path = self.output_dir / filename
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._current_path), fourcc, self.fps,
                (self.width, self.height)
            )

            if not self._writer.isOpened():
                print(f"[Recorder] ファイルを開けません: {self._current_path}")
                self._writer = None
                return None

            self._recording = True
            self._frame_count = 0
            self._start_time = time.time()
            self._preview_buffer.clear()
            self._fullres_buffer.clear()
            self._growing_in = 0
            self._growing_out = -1
            print(f"[Recorder] REC開始: {self._current_path}")
            return self._current_path

    def write_frame(self, frame):
        """フレームを書き込み + バッファ蓄積"""
        with self._lock:
            if not self._recording or self._writer is None:
                return
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self._writer.write(frame)
            self._frame_count += 1

            if len(self._preview_buffer) < self.MAX_BUFFER_FRAMES:
                # プレビュー用縮小フレーム
                small = cv2.resize(frame, (_PREVIEW_W, _PREVIEW_H))
                self._preview_buffer.append(small)
                # フルレスJPEG (クリップ切り出し用)
                _, jpg = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                self._fullres_buffer.append(jpg.tobytes())

    def get_buffered_frame(self, index):
        """バッファからプレビューフレームを取得 (縮小サイズ)"""
        with self._lock:
            if 0 <= index < len(self._preview_buffer):
                return self._preview_buffer[index].copy()
            return None

    def export_clip(self, in_frame, out_frame, output_path):
        """グローウィングバッファからクリップを切り出してMP4書き出し

        録画中でも呼べる。バッファ内のJPEGフレームをデコードして書き出す。
        Returns: 書き出し成功時はフレーム数、失敗時は0
        """
        with self._lock:
            buf_len = len(self._fullres_buffer)
            if buf_len == 0:
                return 0
            in_f = max(0, in_frame)
            out_f = min(out_frame, buf_len - 1) if out_frame >= 0 else buf_len - 1
            if in_f > out_f:
                return 0
            # バッファ参照をコピー (ロック時間を最小化)
            frames_to_write = self._fullres_buffer[in_f:out_f + 1]

        # ロック外でデコード＆書き出し
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps,
            (self.width, self.height))
        if not writer.isOpened():
            print(f"[Recorder] クリップ書き出し失敗: {output_path}")
            return 0

        count = 0
        for jpg_bytes in frames_to_write:
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                writer.write(frame)
                count += 1

        writer.release()
        print(f"[Recorder] クリップ切り出し: {output_path.name} "
              f"({in_f}-{out_f}, {count} frames)")
        return count

    def stop_recording(self):
        """録画停止。録画ファイルパスを返す"""
        with self._lock:
            if not self._recording:
                return None

            self._recording = False
            if self._writer:
                self._writer.release()
                self._writer = None

            path = self._current_path
            duration = self._frame_count / max(self.fps, 1)
            print(f"[Recorder] REC停止: {path} ({self._frame_count} frames, {duration:.1f}s)")
            return path
