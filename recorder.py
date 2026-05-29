"""
録画モジュール

DeckLinkまたはフォールバック入力からの映像をファイルに記録する。
グローウィング対応: 録画中のフレームをJPEGバッファに保持し、
録画しながらスクラブ/In/Out/クリップ切り出しが可能。
"""

import collections
import datetime
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ffmpeg_writer import FFmpegWriter, find_ffmpeg, ffmpeg_extract


# プレビュー用縮小解像度
_PREVIEW_W = 480
_PREVIEW_H = 270

# JPEG圧縮品質 (フルレスバッファ用 — 録画後に高品質で再書き出しするため低めでOK)
_JPEG_QUALITY = 75


class Recorder:
    """入力映像のRECORD/STOP制御 + グローウィングバッファ"""

    def __init__(self, output_dir: str, width=1920, height=1080, fps=29.97,
                 crf=18, growing_buffer_sec=60):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.max_buffer_frames = int(growing_buffer_sec * max(fps, 1))

        self._recording = False
        self._writer = None
        self._current_path = None
        self._frame_count = 0
        self._start_time = None
        self._lock = threading.Lock()

        # グローウィングバッファ (リングバッファ, deque で O(1) 先頭削除)
        self._preview_buffer = collections.deque()   # 縮小フレーム (numpy, 480x270)
        self._fullres_buffer = collections.deque()   # JPEG bytes - クリップ切り出し用
        self._buffer_start = 0      # buffer[0] が対応する絶対フレーム番号
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
        """バッファ末尾の絶対フレーム番号+1 (スライダーの to 値に使用)"""
        return self._buffer_start + len(self._preview_buffer)

    @property
    def buffer_start(self):
        """バッファ先頭の絶対フレーム番号 (スライダーの from_ 値に使用)"""
        return self._buffer_start

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
                ts = time.strftime("%m%d_%H%M%S")
                filename = f"rec_{ts}.mp4"

            today_dir = self.output_dir / datetime.date.today().strftime("%m-%d")
            today_dir.mkdir(parents=True, exist_ok=True)
            self._current_path = today_dir / filename
            try:
                self._writer = FFmpegWriter(
                    str(self._current_path), self.width, self.height, self.fps,
                    crf=self.crf, preset="ultrafast")
            except FileNotFoundError:
                # ffmpeg が無い場合は cv2 にフォールバック
                print("[Recorder] ffmpeg未検出、cv2.VideoWriter にフォールバック")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self._current_path), fourcc, self.fps,
                    (self.width, self.height))

            if not self._writer.isOpened():
                print(f"[Recorder] ファイルを開けません: {self._current_path}")
                self._writer = None
                return None

            self._recording = True
            self._frame_count = 0
            self._start_time = time.time()
            self._preview_buffer.clear()
            self._fullres_buffer.clear()
            self._buffer_start = 0
            self._growing_in = 0
            self._growing_out = -1
            print(f"[Recorder] REC開始: {self._current_path}")
            return self._current_path

    def write_frame(self, frame):
        """フレームを書き込み + バッファ蓄積"""
        with self._lock:
            if not self._recording or self._writer is None:
                return
            if not self._writer.isOpened():
                print("[Recorder] VideoWriter is not opened, stopping")
                self._recording = False
                return
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self._writer.write(frame)
            self._frame_count += 1

            # プレビュー用縮小フレーム
            small = cv2.resize(frame, (_PREVIEW_W, _PREVIEW_H))
            self._preview_buffer.append(small)
            # バッファ用JPEG (プレビュー解像度 — 最終品質は re_export_clip で確保)
            _, jpg = cv2.imencode(
                '.jpg', small,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
            self._fullres_buffer.append(jpg.tobytes())
            # リングバッファ: 上限超過時は古いフレームを破棄
            if len(self._preview_buffer) > self.max_buffer_frames:
                self._preview_buffer.popleft()
                self._fullres_buffer.popleft()
                self._buffer_start += 1

    def get_buffered_frame(self, index):
        """バッファからプレビューフレームを取得 (縮小サイズ)

        index: 絶対フレーム番号 (録画開始からの通し番号)
        """
        with self._lock:
            buf_idx = index - self._buffer_start
            if 0 <= buf_idx < len(self._preview_buffer):
                return self._preview_buffer[buf_idx].copy()
            return None

    def export_clip(self, in_frame, out_frame, output_path):
        """グローウィングバッファからクリップを切り出してMP4書き出し

        録画中でも呼べる。バッファ内のJPEGフレームをデコードして書き出す。
        in_frame, out_frame: 絶対フレーム番号
        Returns: 書き出し成功時はフレーム数、失敗時は0
        """
        with self._lock:
            buf_len = len(self._fullres_buffer)
            if buf_len == 0:
                return 0
            buf_end = self._buffer_start + buf_len  # バッファ末尾+1の絶対番号
            in_f = max(self._buffer_start, in_frame)
            out_f = min(out_frame, buf_end - 1) if out_frame >= 0 else buf_end - 1
            if in_f > out_f:
                return 0
            # 絶対→相対インデックスに変換してコピー (deque はスライス不可)
            rel_in = in_f - self._buffer_start
            rel_out = out_f - self._buffer_start
            frames_to_write = [self._fullres_buffer[i]
                               for i in range(rel_in, rel_out + 1)]

        # ロック外でデコード＆書き出し
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = FFmpegWriter(
                str(output_path), self.width, self.height, self.fps,
                crf=self.crf, preset="fast")
        except FileNotFoundError:
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

    @staticmethod
    def re_export_clip(rec_path, in_frame, out_frame, output_path, fps,
                       crf=18, hw_encode=False):
        """REC本体から指定範囲を再書き出し (録画停止後に呼ぶ)

        グローウィング切り出しで作られたJPEG経由クリップを
        REC本体から直接再エンコードして差し替える。
        Returns: 成功時 True
        """
        if not Path(rec_path).exists():
            print(f"[Recorder] 再書き出し元が見つかりません: {rec_path}")
            return False
        ok = ffmpeg_extract(rec_path, output_path, in_frame, out_frame, fps,
                            crf=crf, preset="medium", hw_encode=hw_encode)
        if ok:
            print(f"[Recorder] 高品質再書き出し完了: {Path(output_path).name}")
        else:
            print(f"[Recorder] 再書き出し失敗: {Path(output_path).name}")
        return ok
