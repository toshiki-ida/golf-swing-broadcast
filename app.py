"""
Golf Swing Broadcast System - メインアプリケーション

プロ放送向けゴルフスイング軌道オーバーレイシステム。
DeckLink入出力、録画、In/Out編集、軌道描画、送出を統合管理する。

ページ構成:
  1. 収録 (Capture)   : DeckLink入力のプレビュー、REC/STOP
  2. クリップ (Clips)  : 録画リスト、In/Out設定、トリム書き出し
  3. 編集 (Edit)       : 軌道描画エディタ
  4. 送出 (Playout)    : DeckLink出力送出
  5. 設定 (Settings)   : デバイス選択、フォルダ設定

使い方:
    python app.py
    python app.py --project D:/golf_project
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from tkinter import colorchooser, filedialog, Canvas, PanedWindow

# mp4vコーデックのマルチスレッドデコードでクラッシュする問題の対策
# (Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:173)
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("playout").setLevel(logging.DEBUG)
logging.getLogger("shuttle").setLevel(logging.INFO)

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from decklink_io import DeckLinkInput, DeckLinkOutput, enumerate_decklink_devices
from field_processor import CaptureMode, CAPTURE_MODE_LABELS, CAPTURE_MODE_EFFECTIVE_FPS
from recorder import Recorder
from clip_manager import ClipManager, ClipData, TrajectoryData
from trajectory import (
    TimedSpline, hex_to_bgr, lerp_color_bgr,
    draw_gradient_trail, draw_markers, render_trajectory_on_frame,
)
from playout import PlayoutEngine
from shuttle_pro import ShuttlePRO

# ShuttlePRO v2 ボタンアクション定義
SHUTTLE_ACTIONS = [
    ("none",         "なし"),
    ("play_pause",   "PLAY/PAUSE"),
    ("play",         "PLAY"),
    ("stop",         "STOP"),
    ("cue",          "CUE (頭出し)"),
    ("prev",         "前クリップ"),
    ("next",         "次クリップ"),
    ("speed_1x",     "速度 1x"),
    ("speed_1_2",    "速度 1/2"),
    ("speed_1_4",    "速度 1/4"),
    ("speed_1_8",    "速度 1/8"),
    ("frame_fwd_1",  "+1F"),
    ("frame_back_1", "-1F"),
    ("frame_fwd_5",  "+5F"),
    ("frame_back_5", "-5F"),
]
SHUTTLE_ACTION_KEYS = [a[0] for a in SHUTTLE_ACTIONS]
SHUTTLE_ACTION_LABELS = [a[1] for a in SHUTTLE_ACTIONS]

DEFAULT_SHUTTLE_BUTTONS = {
    "1": "prev", "2": "cue", "3": "play_pause", "4": "next",
    "5": "speed_1x", "6": "speed_1_2", "7": "speed_1_4", "8": "speed_1_8",
    "9": "stop",
}


# =============================================================================
# 設定
# =============================================================================
DEFAULT_PROJECT_DIR = str(Path.home() / "GolfSwingBroadcast")
SPLINE_RESOLUTION = 300
MARKER_RADIUS = 6
POINT_GRAB_RADIUS = 20

GRADIENT_PRESETS = [
    ("#FFFF00", "#FF0000"),
    ("#00FFFF", "#0000FF"),
    ("#00FF00", "#FF8C00"),
    ("#FF00FF", "#800080"),
]


def _compute_fade_alpha(swing, current_frame, base_alpha=0.85):
    """軌道のフェードイン/アウトを考慮した実効アルファを返す。

    fade_frames=0 の場合は end_frame を過ぎたら 0、それ以外は base_alpha。
    fade_frames>0 の場合、最初の点出現から fade_frames かけてフェードイン、
    終了基準フレーム (end_frame があればそれ、なければ最後の点) 付近で fade_frames かけてフェードアウト。
    """
    fade = getattr(swing, "fade_frames", 0)
    end_f = getattr(swing, "end_frame", -1)
    if not swing.points:
        return 0.0

    first_f = swing.points[0][2]
    last_f = swing.points[-1][2]
    end_ref = end_f if end_f >= 0 else last_f

    if fade <= 0:
        # 従来挙動: end_frame を過ぎたら非表示
        if end_f >= 0 and current_frame > end_f:
            return 0.0
        return base_alpha

    # フェードイン (first_f から fade フレーム)
    if current_frame < first_f:
        fi_ratio = 0.0
    else:
        fi_ratio = min(1.0, (current_frame - first_f) / fade)

    # フェードアウト (end_ref の fade フレーム前から end_ref+fade まで)
    # end_ref を超えたらフェードアウト中
    if current_frame <= end_ref:
        fo_ratio = 1.0
    else:
        fo_ratio = max(0.0, 1.0 - (current_frame - end_ref) / fade)

    return base_alpha * min(fi_ratio, fo_ratio)


# =============================================================================
# フレームキャッシュ
# =============================================================================
class FrameCache:
    def __init__(self, video_path, in_frame=0, out_frame=-1):
        self.frames = []
        self._total_expected = 0
        self._loading = False
        self._video_path = str(video_path)
        self._in_frame = in_frame

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if out_frame < 0:
            out_frame = total - 1
        self._out_frame = out_frame
        self._total_expected = min(out_frame + 1, total) - in_frame

        # 最初のフレームだけ同期で読み込み (即表示用)
        cap.set(cv2.CAP_PROP_POS_FRAMES, in_frame)
        ret, f = cap.read()
        if ret:
            self.frames.append(f)
        cap.release()

    def load_remaining(self, on_done=None):
        """残りフレームをバックグラウンドで読み込み"""
        if self._loading or self._total_expected <= 1:
            if on_done:
                on_done()
            return
        self._loading = True

        def _load():
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self._loading = False
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._in_frame + 1)
            for i in range(self._in_frame + 1, self._in_frame + self._total_expected):
                ret, f = cap.read()
                if not ret:
                    break
                self.frames.append(f)
            cap.release()
            self._loading = False
            if on_done:
                on_done()

        threading.Thread(target=_load, daemon=True).start()

    def __len__(self):
        return self._total_expected if self._total_expected > 0 else len(self.frames)

    def __getitem__(self, idx):
        if 0 <= idx < len(self.frames):
            return self.frames[idx].copy()
        return None


# =============================================================================
# アプリ設定の永続化
# =============================================================================
class AppSettings:
    def __init__(self, project_dir):
        self.path = Path(project_dir) / "settings.json"
        self.data = {
            "project_dir": str(project_dir),
            "record_dir": str(Path(project_dir) / "recordings"),
            "input_device": 0,
            "output_device": 0,
            "width": 1920,
            "height": 1080,
            "fps": 29.97,
            "capture_mode": "normal",
            "shuttle_buttons": dict(DEFAULT_SHUTTLE_BUTTONS),
            "playout_sash_x": 0,   # 0=未設定 (デフォルト width 使用)
            "clips_sash_x": 0,     # 0=未設定
            "edit_sash_x": 0,      # 0=未設定
        }
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.data.update(saved)
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


# =============================================================================
# cv2フレーム → PhotoImage 変換
# =============================================================================
def frame_to_photo(frame, max_w, max_h):
    """OpenCVフレームをTkinter表示用に変換

    アスペクト比を維持して、キャンバスに収まる最大サイズに拡縮する。
    拡大もOK (映像全体が切れずに表示される)。
    """
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb)), 1.0
    scale = min(max_w / w, max_h / h)
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img), scale


# =============================================================================
# メインアプリ
# =============================================================================
class GolfBroadcastApp(ctk.CTk):
    def __init__(self, project_dir=None):
        super().__init__()

        self.title("Golf Swing Broadcast System")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # プロジェクト
        self.project_dir = Path(project_dir or DEFAULT_PROJECT_DIR)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.settings = AppSettings(self.project_dir)
        self.clip_manager = ClipManager(str(self.project_dir))

        # デバイス
        self.deck_input = None
        self.deck_output = None
        self.recorder = Recorder(
            self.settings["record_dir"],
            self.settings["width"],
            self.settings["height"],
            self.settings["fps"],
        )

        # 送出エンジン
        self.playout = PlayoutEngine()
        self._playout_json = str(self.project_dir / "playout.json")
        self._exports_dir = self.project_dir / "exports"
        self._exports_dir.mkdir(parents=True, exist_ok=True)
        self.playout.load_playlist(self._playout_json)
        self.playout.scan_directory(self._exports_dir)

        # 録画書き込みキュー
        # DeckLink COMコールバックスレッドを録画I/Oから分離し、カクツキを防止する。
        # コールバックはフレームをキューに入れるだけ (< 1ms)、
        # 別スレッドが MP4書き込み + JPEGエンコードを担当する。
        self._capture_queue: queue.Queue = queue.Queue(maxsize=1800)
        self._capture_dropped = 0
        self._capture_write_running = True
        self._capture_write_thread = threading.Thread(
            target=self._capture_write_loop, daemon=True, name="CaptureWriteThread"
        )
        self._capture_write_thread.start()

        # ウィンドウサイズ
        self.geometry("1400x900")
        self.minsize(1200, 700)

        # フレーム送りステップ (編集/送出タブ共通)
        self._frame_step = 1

        # UI
        self._build_ui()
        self._bind_global_keys()
        self._refresh_playout_list()

        # ShuttlePRO v2
        self.shuttle = ShuttlePRO()
        self._setup_shuttle()
        self.shuttle.start()

        # 終了時処理
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # キャプチャ自動開始 (UIレイアウト完了後)
        self.after(500, self._start_capture)

    # =========================================================================
    # UI構築
    # =========================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # タブビュー
        self.tabview = ctk.CTkTabview(self, segmented_button_selected_color="#1a6b1a")
        self.tabview.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.tab_capture = self.tabview.add("収録")
        self.tab_clips = self.tabview.add("クリップ")
        self.tab_edit = self.tabview.add("編集")
        self.tab_playout = self.tabview.add("送出")
        self.tab_settings = self.tabview.add("設定")

        self._build_capture_tab()
        self._build_clips_tab()
        self._build_edit_tab()
        self._build_playout_tab()
        self._build_settings_tab()

        self.tabview.configure(command=self._on_tab_changed)

    def _on_tab_changed(self):
        """タブ切り替え時の処理"""
        current = self.tabview.get()
        if current == "クリップ":
            self._refresh_clips_list()
        elif current == "編集":
            self._sync_folder_clips()
            self._refresh_edit_clips_list()
        elif current == "送出":
            # 編集中の軌道を自動保存
            self._edit_autosave_trajectory()
            # exportsディレクトリから新規ファイルを自動取り込み
            added = self.playout.scan_directory(self._exports_dir)
            if added:
                self.playout.save_playlist(self._playout_json)
            self._refresh_playout_list()

    # =========================================================================
    # 収録タブ
    # =========================================================================
    def _build_capture_tab(self):
        tab = self.tab_capture
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # プレビュー
        self.capture_canvas = Canvas(tab, bg="black", highlightthickness=0)
        self.capture_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._capture_photo = None

        # コントロール
        ctrl = ctk.CTkFrame(tab)
        ctrl.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.rec_btn = ctk.CTkButton(
            ctrl, text="⏺ REC", width=120, height=50,
            font=("", 18, "bold"),
            fg_color="#8B0000", hover_color="#B22222",
            command=self._toggle_rec
        )
        self.rec_btn.pack(side="left", padx=10, pady=5)

        self.rec_status = ctk.CTkLabel(ctrl, text="STANDBY", font=("", 14))
        self.rec_status.pack(side="left", padx=20)

        self.rec_time_label = ctk.CTkLabel(ctrl, text="00:00:00", font=("", 20, "bold"))
        self.rec_time_label.pack(side="left", padx=20)

        ctk.CTkButton(ctrl, text="入力開始", width=100,
                       command=self._start_capture).pack(side="right", padx=10)
        ctk.CTkButton(ctrl, text="入力停止", width=100,
                       command=self._stop_capture).pack(side="right", padx=5)

        # 入力モード選択
        mode_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        mode_frame.pack(side="left", padx=15)

        ctk.CTkLabel(mode_frame, text="入力モード:", font=("", 12)).pack(
            side="left", padx=(0, 6))

        saved_mode = self.settings.data.get("capture_mode", "normal")
        mode_values = list(CAPTURE_MODE_LABELS.values())
        self._capture_mode_seg = ctk.CTkSegmentedButton(
            mode_frame,
            values=mode_values,
            command=self._on_capture_mode_changed,
        )
        default_label = CAPTURE_MODE_LABELS.get(
            CaptureMode(saved_mode), CAPTURE_MODE_LABELS[CaptureMode.Normal]
        )
        self._capture_mode_seg.set(default_label)
        self._capture_mode_seg.pack(side="left")

        self._fps_display = ctk.CTkLabel(
            mode_frame,
            text=f"実効: {CAPTURE_MODE_EFFECTIVE_FPS[CaptureMode(saved_mode)]:.2f} fps",
            font=("", 11),
            text_color="#888888",
        )
        self._fps_display.pack(side="left", padx=(8, 0))

        # タイマー更新
        self._update_capture_timer_id = None

    def _get_current_capture_mode(self) -> CaptureMode:
        """現在選択中のキャプチャモードを返す"""
        if hasattr(self, "_capture_mode_seg"):
            label = self._capture_mode_seg.get()
            for mode, lbl in CAPTURE_MODE_LABELS.items():
                if lbl == label:
                    return mode
        saved = self.settings.data.get("capture_mode", "normal")
        return CaptureMode(saved)

    def _on_capture_mode_changed(self, value: str):
        """モード切替コールバック: DeckLinkに即時反映"""
        mode = CaptureMode.Normal
        for m, lbl in CAPTURE_MODE_LABELS.items():
            if lbl == value:
                mode = m
                break
        # 設定保存
        self.settings["capture_mode"] = mode.value
        # 実効fps表示更新
        fps = CAPTURE_MODE_EFFECTIVE_FPS[mode]
        if hasattr(self, "_fps_display"):
            self._fps_display.configure(text=f"実効: {fps:.2f} fps")
        # キャプチャ中なら即時反映
        if self.deck_input:
            self.deck_input.capture_mode = mode
        print(f"[App] キャプチャモード変更: {mode.value} (実効fps={fps:.2f})")

    def _start_capture(self):
        """DeckLink/カメラ入力開始"""
        if self.deck_input:
            self.deck_input.stop()

        mode = self._get_current_capture_mode()
        self.deck_input = DeckLinkInput(
            self.settings["input_device"],
            self.settings["width"],
            self.settings["height"],
            self.settings["fps"],
            capture_mode=mode,
        )
        try:
            self.deck_input.start(frame_callback=self._on_capture_frame)
        except Exception as e:
            print(f"[Capture] 入力開始エラー: {e}")
        self._start_capture_preview()

    def _stop_capture(self):
        """入力停止"""
        if self.deck_input:
            self.deck_input.stop()
            self.deck_input = None

    def _on_capture_frame(self, frame):
        """キャプチャフレーム受信コールバック (DeckLink COMスレッドから呼ばれる)

        このメソッドはできるだけ早く返す必要がある。
        録画中はフレームをキューに追加するだけにし、
        実際のディスク書き込みは _capture_write_loop に委譲する。

        通常モード: bob deinterlace 2倍出力の半分を間引いて29.97fpsで録画
        HFRモード:  全フレーム録画 (29.97fps記録 → スローモーション)
        """
        if self.recorder.is_recording and not getattr(self, '_rec_stopping', False):
            # フレーム間引き (通常モード: 2フレーム中1つだけ録画)
            divisor = getattr(self.deck_input, 'recording_frame_divisor', 1) if self.deck_input else 1
            if divisor > 1:
                self._rec_frame_count = getattr(self, '_rec_frame_count', 0) + 1
                if self._rec_frame_count % divisor != 0:
                    return
            try:
                self._capture_queue.put_nowait(frame)
            except queue.Full:
                self._capture_dropped += 1  # キューが満杯 = フレームドロップ (ディスクが遅い場合)

    def _capture_write_loop(self):
        """録画書き込みスレッド: キューからフレームを取り出してディスクに書き込む"""
        while self._capture_write_running:
            try:
                frame = self._capture_queue.get(timeout=0.1)
                self.recorder.write_frame(frame)
                self._capture_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[CaptureWrite] エラー: {e}")

    def _start_capture_preview(self):
        """キャプチャプレビュー更新ループ"""
        if self._update_capture_timer_id:
            self.after_cancel(self._update_capture_timer_id)

        self._no_signal_shown = False

        def update():
            if self.deck_input:
                frame = self.deck_input.get_frame()
                cw = self.capture_canvas.winfo_width()
                ch = self.capture_canvas.winfo_height()

                if frame is not None:
                    self._no_signal_shown = False
                    if cw > 10 and ch > 10:
                        self._capture_photo, _ = frame_to_photo(frame, cw, ch)
                        self.capture_canvas.delete("all")
                        self.capture_canvas.create_image(
                            cw // 2, ch // 2, anchor="center", image=self._capture_photo)
                elif not self._no_signal_shown and cw > 10 and ch > 10:
                    # フレーム未到着: NO SIGNAL 表示
                    self._no_signal_shown = True
                    self.capture_canvas.delete("all")
                    self.capture_canvas.create_text(
                        cw // 2, ch // 2 - 15, text="NO SIGNAL",
                        fill="#888888", font=("", 32, "bold"), anchor="center")
                    self.capture_canvas.create_text(
                        cw // 2, ch // 2 + 25, text="入力信号を待機中...",
                        fill="#666666", font=("", 14), anchor="center")

                # REC状態更新
                if self.recorder.is_recording:
                    sec = self.recorder.duration_sec
                    h = int(sec // 3600)
                    m = int((sec % 3600) // 60)
                    s = int(sec % 60)
                    self.rec_time_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
                    # グローウィングクリップのラベル更新
                    if self._rec_live_label:
                        fc = self.recorder.frame_count
                        self._rec_live_label.configure(
                            text=f"● REC {m:02d}:{s:02d} ({fc}f)")
                    # グローウィングクリッププレビュー更新
                    if (self._selected_clip_id == "__growing__"
                            and self._growing_follow_live):
                        buf_cnt = self.recorder.buffered_frame_count
                        if buf_cnt > 0:
                            self.clip_slider.configure(to=max(buf_cnt - 1, 1))
                            self.clip_slider.set(buf_cnt - 1)
                            self._show_growing_preview(buf_cnt - 1)

            self._update_capture_timer_id = self.after(33, update)

        update()

    def _toggle_rec(self):
        """REC/STOP切り替え"""
        if self.recorder.is_recording:
            # グローウィング中にセットされたIn/Outを保持
            g_in = self.recorder.growing_in
            g_out = self.recorder.growing_out
            # 新規フレームのキュー投入を停止 (コールバック側で判定)
            self._rec_stopping = True
            # キューに残っているフレームを全て書き出してから停止
            # (即座にstopするとVideoWriter.release()でmoov atomが不完全になる)
            try:
                self._capture_queue.join()
            except Exception:
                pass
            self._rec_stopping = False
            frames = self.recorder.frame_count
            if self._capture_dropped > 0:
                print(f"[Capture] 録画中にドロップしたフレーム数: {self._capture_dropped}")
            path = self.recorder.stop_recording()
            self.rec_btn.configure(text="⏺ REC", fg_color="#8B0000")
            self.rec_status.configure(text="STANDBY", text_color="white")
            if path and frames > 0:
                self.after(500, lambda p=str(path), gi=g_in, go=g_out:
                           self._add_recorded_clip(p, growing_in=gi, growing_out=go))
            elif path and frames == 0:
                print(f"[Capture] 0フレーム録画のためスキップ: {path}")
            self._refresh_clips_list()
        else:
            if not self.deck_input:
                self._start_capture()
            # 録画fps: 通常=29.97fps(等倍再生), HFR=29.97fps(スロー)
            rec_fps = self.deck_input.recording_fps if self.deck_input else self.settings["fps"]
            self._rec_frame_count = 0  # フレーム間引きカウンタリセット
            self._capture_dropped = 0  # ドロップカウンタリセット
            self.recorder.start_recording(fps=rec_fps)
            fps_label = f" ({rec_fps:.2f}fps)" if rec_fps != self.settings["fps"] else ""
            self.rec_btn.configure(text="⏹ STOP", fg_color="#FF0000")
            self.rec_status.configure(text=f"● REC{fps_label}", text_color="red")
            self._refresh_clips_list()

    def _add_recorded_clip(self, path, retries=3, growing_in=0, growing_out=-1):
        """録画ファイルをクリップに追加 (リトライ付き、グローウィングIn/Out引き継ぎ)"""
        try:
            # フォルダスキャンで既に登録済みなら再追加しない
            resolved = str(Path(path).resolve())
            clip = next((c for c in self.clip_manager.clips
                         if str(Path(c.source_path).resolve()) == resolved), None)
            if clip is None:
                clip = self.clip_manager.add_clip(path)
            # グローウィング中に設定されたIn/Outを引き継ぎ
            if growing_in > 0 or growing_out >= 0:
                in_f = growing_in
                out_f = growing_out if growing_out >= 0 else clip.total_frames - 1
                self.clip_manager.set_in_out(clip.id, in_f, out_f)
                print(f"[Capture] グローウィングIn/Out引き継ぎ: {in_f}-{out_f}")
            self._refresh_clips_list()
            print(f"[Capture] クリップ追加: {path}")
        except Exception as e:
            if retries > 0:
                print(f"[Capture] クリップ追加リトライ ({retries}回残り): {e}")
                self.after(500, lambda: self._add_recorded_clip(
                    path, retries - 1, growing_in, growing_out))
            else:
                print(f"[Capture] クリップ追加失敗: {e}")

    # =========================================================================
    # クリップタブ
    # =========================================================================
    def _build_clips_tab(self):
        tab = self.tab_clips
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # PanedWindow でリサイズ可能な左右分割
        self.clips_paned = PanedWindow(
            tab, orient="horizontal", sashwidth=6,
            bg="#2b2b2b", sashrelief="flat", borderwidth=0,
        )
        self.clips_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # リスト (左)
        list_frame = ctk.CTkFrame(self.clips_paned, width=300)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        # ツールバー
        toolbar = ctk.CTkFrame(list_frame, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ctk.CTkButton(toolbar, text="ファイル追加", width=120,
                       command=self._add_clip_from_file).pack(side="left", padx=5)
        ctk.CTkButton(toolbar, text="削除", width=80,
                       fg_color="#8B0000", hover_color="#A52A2A",
                       command=self._delete_selected_clip).pack(side="left", padx=5)
        ctk.CTkButton(toolbar, text="更新", width=80,
                       command=self._refresh_clips_list).pack(side="left", padx=5)

        # クリップリスト（スクロール可能）
        self.clips_scroll = ctk.CTkScrollableFrame(list_frame)
        self.clips_scroll.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.clips_scroll.grid_columnconfigure(0, weight=1)
        self.clip_widgets = []
        self._selected_clip_id = None
        self._rec_live_row = None
        self._rec_live_label = None
        self._growing_follow_live = True  # ライブ追従モード

        # 右パネル
        right = ctk.CTkFrame(self.clips_paned)

        ctk.CTkLabel(right, text="クリップ情報", font=("", 16, "bold")).pack(pady=10)

        self.clip_info_label = ctk.CTkLabel(right, text="クリップを選択してください",
                                             wraplength=300)
        self.clip_info_label.pack(padx=10, pady=5)

        # プレビュー (可能な限り大きく)
        self.clip_preview_canvas = Canvas(right, bg="black", highlightthickness=0)
        self.clip_preview_canvas.pack(fill="both", expand=True, padx=10, pady=5)
        self._clip_preview_photo = None

        # 名称変更
        name_frame = ctk.CTkFrame(right, fg_color="transparent")
        name_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(name_frame, text="名前:").pack(side="left")
        self.clip_name_entry = ctk.CTkEntry(name_frame, width=180)
        self.clip_name_entry.pack(side="left", padx=5)
        self.clip_name_entry.bind("<Return>", lambda e: self._rename_clip())
        ctk.CTkButton(name_frame, text="変更", width=60,
                       command=self._rename_clip).pack(side="left", padx=2)

        # In/Out表示
        io_frame = ctk.CTkFrame(right, fg_color="transparent")
        io_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(io_frame, text="In:").pack(side="left")
        self.in_entry = ctk.CTkEntry(io_frame, width=70)
        self.in_entry.pack(side="left", padx=3)
        ctk.CTkButton(io_frame, text="↓", width=30,
                       command=self._set_in_current).pack(side="left")
        ctk.CTkLabel(io_frame, text="Out:").pack(side="left", padx=(8, 0))
        self.out_entry = ctk.CTkEntry(io_frame, width=70)
        self.out_entry.pack(side="left", padx=3)
        ctk.CTkButton(io_frame, text="↓", width=30,
                       command=self._set_out_current).pack(side="left")

        # スライダー
        self.clip_slider = ctk.CTkSlider(right, from_=0, to=100,
                                          command=self._on_clip_slider)
        self.clip_slider.pack(fill="x", padx=10, pady=5)
        self._clip_slider_frame = 0

        # グローウィングクリップ切り出しボタン
        self.clip_extract_btn = ctk.CTkButton(
            right, text="クリップ切り出し", width=250, height=40,
            font=("", 14, "bold"),
            fg_color="#B8860B", hover_color="#DAA520",
            command=self._extract_growing_clip,
        )
        self.clip_extract_btn.pack(padx=10, pady=(5, 0))
        self.clip_extract_btn.pack_forget()  # 初期非表示

        # PanedWindow に左右を追加 (最小幅を指定)
        self.clips_paned.add(list_frame, minsize=200, stretch="never")
        self.clips_paned.add(right, minsize=400, stretch="always")

        # 保存されたサッシ位置を復元
        self.clips_paned.bind("<Configure>", self._clips_paned_configure)
        self.clips_paned.bind("<ButtonRelease-1>", self._save_clips_sash)
        self._clips_sash_restored = False

        # プレビューキャンバスのリサイズ対応
        self.clip_preview_canvas.bind("<Configure>", self._on_clip_preview_resize)
        self._clip_preview_resize_after = None

        self._refresh_clips_list()

    def _clips_paned_configure(self, event):
        """PanedWindow初回表示時にサッシ位置を復元"""
        if self._clips_sash_restored:
            return
        saved_x = self.settings.data.get("clips_sash_x", 0)
        total_w = self.clips_paned.winfo_width()
        if saved_x > 0 and total_w > saved_x + 50:
            try:
                self.clips_paned.sash_place(0, saved_x, 0)
                self._clips_sash_restored = True
            except Exception:
                pass
        elif total_w > 400:
            try:
                self.clips_paned.sash_place(0, 300, 0)
                self._clips_sash_restored = True
            except Exception:
                pass

    def _save_clips_sash(self, event=None):
        try:
            coord = self.clips_paned.sash_coord(0)
            self.settings["clips_sash_x"] = coord[0]
            self.settings.save()
        except Exception:
            pass

    def _on_clip_preview_resize(self, event=None):
        """プレビューキャンバスリサイズ時に再描画 (debounce)"""
        if self._clip_preview_resize_after:
            try:
                self.after_cancel(self._clip_preview_resize_after)
            except Exception:
                pass
        self._clip_preview_resize_after = self.after(
            100, self._redraw_clip_preview)

    def _redraw_clip_preview(self):
        """現在のクリッププレビューを再描画"""
        self._clip_preview_resize_after = None
        cid = getattr(self, "_selected_clip_id", None)
        if not cid:
            return
        try:
            frame_no = getattr(self, "_clip_slider_frame", 0)
            if cid == "__growing__":
                self._show_growing_preview(frame_no)
            else:
                clip = self.clip_manager.get_clip(cid)
                if clip:
                    self._show_clip_preview(clip, frame_no)
        except Exception:
            pass

    def _add_clip_from_file(self):
        paths = filedialog.askopenfilenames(
            title="動画ファイルを選択",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")]
        )
        for p in paths:
            self.clip_manager.add_clip(p)
        self._refresh_clips_list()

    def _sync_folder_clips(self):
        """recordings/ と clips/ フォルダの実ファイルからクリップリストを構築。
        - 実ファイルが無いエントリは除去
        - 同一ファイルの重複エントリは除去
        - フォルダにあって未登録のファイルは追加
        """
        VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
        dirty = False

        # 1) 実ファイルが存在しないクリップを除去
        before = len(self.clip_manager.clips)
        self.clip_manager.clips = [
            c for c in self.clip_manager.clips if Path(c.source_path).exists()
        ]
        if len(self.clip_manager.clips) < before:
            print(f"[Clips] ファイルなし {before - len(self.clip_manager.clips)} 件除去")
            dirty = True

        # 2) 同一source_pathの重複を除去 (最初のエントリを残す)
        seen = set()
        deduped = []
        for c in self.clip_manager.clips:
            key = str(Path(c.source_path).resolve())
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        if len(deduped) < len(self.clip_manager.clips):
            print(f"[Clips] 重複 {len(self.clip_manager.clips) - len(deduped)} 件除去")
            self.clip_manager.clips = deduped
            dirty = True

        if dirty:
            self.clip_manager.save()

        # 3) フォルダ内の未登録ファイルを追加
        scan_dirs = [
            Path(self.settings["record_dir"]),
            self.clip_manager.clips_dir,
        ]
        known_paths = {str(Path(c.source_path).resolve()) for c in self.clip_manager.clips}
        added = 0
        for d in scan_dirs:
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    key = str(f.resolve())
                    if key not in known_paths:
                        try:
                            self.clip_manager.add_clip(str(f))
                            known_paths.add(key)
                            added += 1
                        except Exception as e:
                            print(f"[Clips] スキャン追加エラー: {f.name}: {e}")
        if added:
            print(f"[Clips] フォルダスキャンで {added} 件追加")

    def _refresh_clips_list(self):
        self._sync_folder_clips()
        for w in self.clip_widgets:
            w.destroy()
        self.clip_widgets.clear()
        self._rec_live_row = None
        self._rec_live_label = None

        row_idx = 0

        # --- 録画中エントリ (先頭に赤く表示) ---
        if self.recorder.is_recording:
            row = ctk.CTkFrame(self.clips_scroll, height=40,
                               fg_color="#3D0000", border_color="#FF0000", border_width=1)
            row.grid(row=row_idx, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(row, text="●", width=30, text_color="#FF0000",
                         font=("", 14, "bold")).grid(row=0, column=0, padx=5)

            fname = Path(self.recorder.current_file).stem if self.recorder.current_file else "REC"
            ctk.CTkButton(
                row, text=fname, anchor="w",
                text_color="#FF6666", font=("", 13, "bold"),
                fg_color="transparent", hover_color="#550000",
                command=self._select_growing_clip,
            ).grid(row=0, column=1, sticky="ew", padx=5)

            self._rec_live_label = ctk.CTkLabel(
                row, text="REC 00:00", width=100,
                text_color="#FF0000", font=("", 12, "bold"))
            self._rec_live_label.grid(row=0, column=2, padx=5)

            self._rec_live_row = row
            self.clip_widgets.append(row)
            row_idx += 1

        # --- 保存済みクリップ ---
        for i, clip in enumerate(self.clip_manager.clips):
            row = ctk.CTkFrame(self.clips_scroll, height=40)
            row.grid(row=row_idx, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            # 番号
            ctk.CTkLabel(row, text=f"{i+1}", width=30).grid(row=0, column=0, padx=5)
            # 名前
            name_btn = ctk.CTkButton(
                row, text=clip.name, anchor="w",
                fg_color="transparent", hover_color="#333333",
                command=lambda cid=clip.id: self._select_clip(cid)
            )
            name_btn.grid(row=0, column=1, sticky="ew", padx=5)
            # 情報
            dur = f"{clip.duration_sec:.1f}s"
            traj = "✓" if clip.has_trajectory else ""
            ctk.CTkLabel(row, text=f"{dur}  {traj}", width=100).grid(row=0, column=2, padx=5)

            # 削除ボタン
            ctk.CTkButton(
                row, text="×", width=28, height=28,
                fg_color="#8B0000", hover_color="#A52A2A",
                font=("", 14, "bold"),
                command=lambda cid=clip.id: self._delete_clip_by_id(cid),
            ).grid(row=0, column=3, padx=(0, 5))

            self.clip_widgets.append(row)
            row_idx += 1

    def _select_clip(self, clip_id):
        self._selected_clip_id = clip_id
        clip = self.clip_manager.get_clip(clip_id)
        if not clip:
            return

        self.clip_info_label.configure(
            text=f"{clip.width}x{clip.height} | {clip.fps:.2f}fps\n"
                 f"{clip.total_frames} frames ({clip.duration_sec:.1f}s)"
        )

        self.clip_name_entry.delete(0, "end")
        self.clip_name_entry.insert(0, clip.name)

        self.in_entry.delete(0, "end")
        self.in_entry.insert(0, str(clip.in_frame))
        self.out_entry.delete(0, "end")
        self.out_entry.insert(0, str(clip.get_out_frame()))

        self.clip_slider.configure(to=max(clip.total_frames - 1, 1))
        self.clip_slider.set(clip.in_frame)
        self._show_clip_preview(clip, clip.in_frame)

        # クリップ切り出しボタンを表示
        self.clip_extract_btn.configure(text="クリップ切り出し", state="normal")
        self.clip_extract_btn.pack(padx=10, pady=(5, 0))

    def _show_clip_preview(self, clip, frame_no):
        cap = cv2.VideoCapture(clip.source_path)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cw = self.clip_preview_canvas.winfo_width()
            ch = self.clip_preview_canvas.winfo_height()
            if cw < 10 or ch < 10:
                cw, ch = 320, 180
            self._clip_preview_photo, _ = frame_to_photo(frame, cw, ch)
            self.clip_preview_canvas.delete("all")
            self.clip_preview_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                                   image=self._clip_preview_photo)

    def _select_growing_clip(self):
        """グローウィングクリップを選択"""
        self._selected_clip_id = "__growing__"
        self._growing_follow_live = True
        buf_cnt = self.recorder.buffered_frame_count
        sec = self.recorder.duration_sec

        fname = Path(self.recorder.current_file).stem if self.recorder.current_file else "REC"
        self.clip_info_label.configure(
            text=f"{fname}\n● REC中 (グローウィング)\n"
                 f"{self.recorder.frame_count} frames ({sec:.1f}s)\n"
                 f"In/Outを設定して「クリップ切り出し」"
        )

        self.in_entry.delete(0, "end")
        self.in_entry.insert(0, str(self.recorder.growing_in))
        self.out_entry.delete(0, "end")
        out_val = self.recorder.growing_out
        # buf_cnt==0の場合は0を表示 (まだフレームが溜まっていない)
        out_display = out_val if out_val >= 0 else max(buf_cnt - 1, 0)
        self.out_entry.insert(0, str(out_display))

        slider_max = max(buf_cnt - 1, 1)
        self.clip_slider.configure(to=slider_max)
        self.clip_slider.set(min(max(buf_cnt - 1, 0), slider_max))
        if buf_cnt > 0:
            self._show_growing_preview(buf_cnt - 1)

        # 切り出しボタン表示
        self.clip_extract_btn.pack(padx=10, pady=(5, 0))

    def _show_growing_preview(self, frame_idx):
        """グローウィングバッファからプレビュー表示"""
        frame = self.recorder.get_buffered_frame(frame_idx)
        if frame is not None:
            cw = self.clip_preview_canvas.winfo_width()
            ch = self.clip_preview_canvas.winfo_height()
            if cw < 10 or ch < 10:
                cw, ch = 320, 180
            self._clip_preview_photo, _ = frame_to_photo(frame, cw, ch)
            self.clip_preview_canvas.delete("all")
            self.clip_preview_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                                   image=self._clip_preview_photo)

    def _on_clip_slider(self, value):
        self._clip_slider_frame = int(value)
        if self._selected_clip_id == "__growing__":
            self._growing_follow_live = False  # 手動スクラブ → ライブ追従OFF
            self._show_growing_preview(self._clip_slider_frame)
            return
        clip = self.clip_manager.get_clip(self._selected_clip_id) if self._selected_clip_id else None
        if clip:
            self._show_clip_preview(clip, self._clip_slider_frame)

    def _clip_jump(self, delta):
        """キーボードショートカットによるフレーム送り"""
        if not self._selected_clip_id:
            return
        new_f = self._clip_slider_frame + delta
        # 上限
        if self._selected_clip_id == "__growing__":
            max_f = max(self.recorder.frame_count - 1, 0)
        else:
            clip = self.clip_manager.get_clip(self._selected_clip_id)
            if not clip:
                return
            max_f = max(clip.total_frames - 1, 0)
        new_f = max(0, min(new_f, max_f))
        self._clip_slider_frame = new_f
        try:
            self.clip_slider.set(new_f)
        except Exception:
            pass
        if self._selected_clip_id == "__growing__":
            self._growing_follow_live = False
            self._show_growing_preview(new_f)
        else:
            clip = self.clip_manager.get_clip(self._selected_clip_id)
            if clip:
                self._show_clip_preview(clip, new_f)

    def _set_in_current(self):
        self.in_entry.delete(0, "end")
        self.in_entry.insert(0, str(self._clip_slider_frame))
        if self._selected_clip_id == "__growing__":
            self.recorder.growing_in = self._clip_slider_frame
        else:
            self._apply_in_out()

    def _set_out_current(self):
        self.out_entry.delete(0, "end")
        self.out_entry.insert(0, str(self._clip_slider_frame))
        if self._selected_clip_id == "__growing__":
            self.recorder.growing_out = self._clip_slider_frame
        else:
            self._apply_in_out()

    def _rename_clip(self):
        """クリップ名を変更 (実ファイルも連動してリネーム)

        対象:
          - source_path (録画・収録した元ファイル)
          - exported_path (トリム書き出しファイル)
          - 送出リスト (playlist) 内の同一クリップ参照
        trajectory_path はclip.id由来のためリネーム不要。
        """
        if not self._selected_clip_id or self._selected_clip_id == "__growing__":
            return
        new_name = self.clip_name_entry.get().strip()
        if not new_name:
            return

        # Windowsのファイル名で使えない文字を拒否
        INVALID_CHARS = set('\\/:*?"<>|')
        if any(c in INVALID_CHARS for c in new_name):
            print(f"[Clip] 使用できない文字が含まれています: {new_name}")
            self.clip_info_label.configure(
                text=f"⚠ 使用できない文字:\n\\ / : * ? \" < > |")
            return

        clip = self.clip_manager.get_clip(self._selected_clip_id)
        if not clip:
            return
        if clip.name == new_name:
            return

        old_name = clip.name
        old_source = Path(clip.source_path) if clip.source_path else None
        old_exported = Path(clip.exported_path) if clip.exported_path else None

        # --- リネーム先パスを計算 ---
        new_source = None
        if old_source and old_source.exists():
            new_source = old_source.parent / f"{new_name}{old_source.suffix}"
            if new_source.exists() and new_source.resolve() != old_source.resolve():
                print(f"[Clip] リネーム先が既に存在: {new_source.name}")
                self.clip_info_label.configure(
                    text=f"⚠ 同名ファイルが既に存在します:\n{new_source.name}")
                return

        new_exported = None
        if old_exported and old_exported.exists():
            # stem が old_name で始まる場合は new_name に置換、それ以外は後ろに new_name を付加しない
            old_stem = old_exported.stem
            if old_stem == old_name:
                new_stem = new_name
            elif old_stem.startswith(old_name):
                new_stem = new_name + old_stem[len(old_name):]
            else:
                # 命名規則が違う (例: swing_XXX) → そのまま使う
                new_stem = old_stem
            if new_stem != old_stem:
                new_exported = old_exported.parent / f"{new_stem}{old_exported.suffix}"
                if new_exported.exists() and new_exported.resolve() != old_exported.resolve():
                    print(f"[Clip] エクスポートリネーム先が既に存在: {new_exported.name}")
                    self.clip_info_label.configure(
                        text=f"⚠ 同名ファイルが既に存在します:\n{new_exported.name}")
                    return

        # --- 再生中・キュー中のリソースを解放 (ファイルロック対策) ---
        was_playing = bool(self.playout._playing)
        cap_released = False
        resolved_old = str(old_source.resolve()) if old_source else None
        if resolved_old:
            for item in self.playout.playlist:
                if str(Path(item.clip.source_path).resolve()) == resolved_old:
                    self.playout.stop()
                    # スレッド終了待ち + cap クローズ
                    if self.playout._thread and self.playout._thread.is_alive():
                        self.playout._thread.join(timeout=1.0)
                    self.playout._close_cap()
                    cap_released = True
                    break

        # --- 実ファイルリネーム ---
        renamed_source = False
        try:
            if new_source:
                old_source.rename(new_source)
                renamed_source = True
                clip.source_path = str(new_source.resolve())
            if new_exported:
                try:
                    old_exported.rename(new_exported)
                    clip.exported_path = str(new_exported.resolve())
                except Exception as e:
                    print(f"[Clip] エクスポートファイルのリネームに失敗: {e}")
                    # source はリネーム成功しているので続行
        except Exception as e:
            print(f"[Clip] ファイルリネーム失敗: {e}")
            self.clip_info_label.configure(
                text=f"⚠ リネーム失敗:\n{e}")
            return

        # --- ClipData 更新 ---
        clip.name = new_name
        self.clip_manager.save()

        # --- 送出リスト内の同一クリップ参照も更新 ---
        playlist_updated = False
        for item in self.playout.playlist:
            if item.clip.id == clip.id or (
                    resolved_old and
                    str(Path(item.clip.source_path).resolve()) == resolved_old):
                item.clip.name = new_name
                if renamed_source:
                    item.clip.source_path = clip.source_path
                if new_exported and clip.exported_path:
                    item.clip.exported_path = clip.exported_path
                playlist_updated = True

        if playlist_updated:
            self.playout.save_playlist(self._playout_json)

        # --- UI 更新 ---
        self._refresh_clips_list()
        if hasattr(self, '_refresh_edit_clips_list'):
            self._refresh_edit_clips_list()
        if playlist_updated:
            self._refresh_playout_list()
            # キューし直し (capを開き直し)
            if cap_released and 0 <= self.playout.current_index < len(self.playout.playlist):
                self.playout.cue(self.playout.current_index)

        print(f"[Clip] リネーム: {old_name} → {new_name}")
        self.clip_info_label.configure(
            text=f"リネーム完了:\n{old_name} → {new_name}")

    def _apply_in_out(self):
        if not self._selected_clip_id:
            return
        try:
            in_f = int(self.in_entry.get())
            out_f = int(self.out_entry.get())
        except ValueError:
            return
        if self._selected_clip_id == "__growing__":
            self.recorder.growing_in = in_f
            self.recorder.growing_out = out_f
            return
        self.clip_manager.set_in_out(self._selected_clip_id, in_f, out_f)

    def _export_trim(self):
        if not self._selected_clip_id:
            return
        self._apply_in_out()
        path = self.clip_manager.export_trimmed(self._selected_clip_id)
        if path:
            print(f"トリム書き出し完了: {path}")

    def _extract_growing_clip(self):
        """クリップ切り出し: グローウィング or 通常クリップのIn/Outトリム"""
        # --- グローウィング (録画中) ---
        if self._selected_clip_id == "__growing__" and self.recorder.is_recording:
            try:
                in_f = int(self.in_entry.get())
                out_f = int(self.out_entry.get())
            except ValueError:
                return
            if in_f >= out_f:
                print("[Capture] In/Outが不正です")
                return
            ts = time.strftime("%Y%m%d_%H%M%S")
            clip_name = f"clip_{ts}_{in_f}_{out_f}"
            out_path = self.project_dir / "clips" / f"{clip_name}.mp4"

            def do_export():
                count = self.recorder.export_clip(in_f, out_f, str(out_path))
                if count > 0:
                    self.after(0, lambda: self._finish_extract(str(out_path)))
                else:
                    print(f"[Capture] クリップ切り出し失敗 (0 frames)")

            self.clip_extract_btn.configure(text="切り出し中...", state="disabled")
            threading.Thread(target=do_export, daemon=True).start()
            return

        # --- 通常クリップのIn/Outトリム ---
        if not self._selected_clip_id:
            return
        try:
            in_f = int(self.in_entry.get())
            out_f = int(self.out_entry.get())
        except ValueError:
            return
        if in_f >= out_f:
            print("[Clips] In/Outが不正です")
            return
        self._apply_in_out()
        self.clip_extract_btn.configure(text="切り出し中...", state="disabled")

        clip_id = self._selected_clip_id

        def do_trim():
            path = self.clip_manager.export_trimmed(clip_id)
            if path:
                self.after(0, lambda: self._finish_extract(str(path)))
            else:
                self.after(0, lambda: self.clip_extract_btn.configure(
                    text="クリップ切り出し", state="normal"))
                print("[Clips] トリム失敗")

        threading.Thread(target=do_trim, daemon=True).start()

    def _finish_extract(self, path):
        """切り出し完了後、クリップリストに追加"""
        try:
            clip = self.clip_manager.add_clip(path)
            self._refresh_clips_list()
            # 新しいクリップを選択
            self._select_clip(clip.id)
            print(f"[Capture] グローウィングクリップ追加: {clip.name}")
        except Exception as e:
            print(f"[Capture] クリップ追加エラー: {e}")
        finally:
            self.clip_extract_btn.configure(text="クリップ切り出し", state="normal")

    def _delete_clip_files(self, clip_id):
        """クリップに紐づく実ファイル (source, exported, trajectory) を削除"""
        clip = self.clip_manager.get_clip(clip_id)
        if not clip:
            return
        for p in [clip.source_path, clip.exported_path, clip.trajectory_path]:
            if p:
                fp = Path(p)
                if fp.exists():
                    try:
                        fp.unlink()
                        print(f"[Clips] ファイル削除: {fp.name}")
                    except Exception as e:
                        print(f"[Clips] ファイル削除エラー: {fp.name}: {e}")

    def _delete_selected_clip(self):
        if self._selected_clip_id:
            self._delete_clip_files(self._selected_clip_id)
            self.clip_manager.remove_clip(self._selected_clip_id)
            self._selected_clip_id = None
            self._refresh_clips_list()

    def _delete_clip_by_id(self, clip_id):
        """クリップリストの行ボタンから直接削除"""
        self._delete_clip_files(clip_id)
        self.clip_manager.remove_clip(clip_id)
        if self._selected_clip_id == clip_id:
            self._selected_clip_id = None
        self._refresh_clips_list()

    def _open_edit_for_clip(self):
        if not self._selected_clip_id:
            return
        if self._selected_clip_id == "__growing__":
            self.clip_info_label.configure(
                text="⚠ 録画中は直接編集できません\n"
                     "「クリップ切り出し」で抜き出してから\n"
                     "編集してください")
            return
        # エントリの最新値を clip.in_frame/out_frame に反映
        self._apply_in_out()
        self._edit_clip_id = self._selected_clip_id
        self._load_edit_clip()
        self._refresh_edit_clips_list()
        self.tabview.set("編集")

    # =========================================================================
    # 編集タブ
    # =========================================================================
    def _build_edit_tab(self):
        tab = self.tab_edit
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # PanedWindow でリサイズ可能な左右分割
        self.edit_paned = PanedWindow(
            tab, orient="horizontal", sashwidth=6,
            bg="#2b2b2b", sashrelief="flat", borderwidth=0,
        )
        self.edit_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 左: キャンバス + タイムライン
        left = ctk.CTkFrame(self.edit_paned, fg_color="transparent")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        self.edit_canvas = Canvas(left, bg="black", highlightthickness=0)
        self.edit_canvas.grid(row=0, column=0, sticky="nsew")
        self.edit_canvas.bind("<Button-1>", self._edit_left_click)
        self.edit_canvas.bind("<Button-3>", self._edit_right_press)
        self.edit_canvas.bind("<B3-Motion>", self._edit_right_drag)
        self.edit_canvas.bind("<ButtonRelease-3>", self._edit_right_release)
        self._edit_photo = None

        # タイムライン
        tl_frame = ctk.CTkFrame(left, fg_color="transparent")
        tl_frame.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        tl_frame.grid_columnconfigure(1, weight=1)

        self.edit_frame_label = ctk.CTkLabel(tl_frame, text="0 / 0", width=120)
        self.edit_frame_label.grid(row=0, column=0, padx=(0, 10))

        self.edit_slider = ctk.CTkSlider(tl_frame, from_=0, to=100,
                                          command=self._on_edit_slider)
        self.edit_slider.grid(row=0, column=1, sticky="ew")

        # 再生ボタン
        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.grid(row=2, column=0, sticky="ew", pady=(3, 0))

        ctk.CTkButton(ctrl, text="◀◀", width=45,
                       command=lambda: self._edit_jump(-self._frame_step * 5)).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="◀", width=45,
                       command=lambda: self._edit_jump(-self._frame_step)).pack(side="left", padx=2)
        self.edit_play_btn = ctk.CTkButton(ctrl, text="▶", width=60,
                                            command=self._edit_toggle_play)
        self.edit_play_btn.pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="▶", width=45,
                       command=lambda: self._edit_jump(self._frame_step)).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="▶▶", width=45,
                       command=lambda: self._edit_jump(self._frame_step * 5)).pack(side="left", padx=2)

        # フレーム送りステップ
        ctk.CTkLabel(ctrl, text="  Step:", font=("", 11)).pack(side="left", padx=(15, 3))
        self.edit_step_seg = ctk.CTkSegmentedButton(
            ctrl, values=["1", "2", "5", "10"],
            width=180, command=self._on_frame_step_change)
        self.edit_step_seg.pack(side="left", padx=2)
        self.edit_step_seg.set(str(self._frame_step))

        # 右: クリップリスト + 軌道編集パネル
        right = ctk.CTkFrame(self.edit_paned, width=280)
        right.pack_propagate(False)

        # --- クリップリスト (上端固定) ---
        ctk.CTkLabel(right, text="クリップ", font=("", 14, "bold")).pack(pady=(8, 2))
        self.edit_clips_scroll = ctk.CTkScrollableFrame(right, height=160)
        self.edit_clips_scroll.pack(fill="x", padx=5, pady=(0, 5))
        self.edit_clips_scroll.grid_columnconfigure(0, weight=1)
        self._edit_clip_widgets = []

        # --- アクションボタン (下端固定, 下から順に pack(side="bottom")) ---
        # 配置順に注意: pack(side="bottom") は先に pack したものが下になる
        ctk.CTkButton(right, text="軌道を削除", width=250, height=30,
                       fg_color="#8B0000", hover_color="#A52A2A",
                       command=self._edit_delete_trajectory).pack(side="bottom", padx=10, pady=(5, 8))

        ctk.CTkButton(right, text="動画書き出し", width=250, height=35,
                       fg_color="#003366", hover_color="#004488",
                       command=self._edit_export_video).pack(side="bottom", padx=10, pady=5)

        ctk.CTkButton(right, text="軌道を保存", width=250, height=40,
                       font=("", 14, "bold"),
                       fg_color="#006400", hover_color="#228B22",
                       command=self._edit_save_trajectory).pack(side="bottom", padx=10, pady=(5, 5))

        # --- スクロール可能な軌道編集エリア (中央、残りのスペースを占有) ---
        edit_scroll = ctk.CTkScrollableFrame(right, fg_color="transparent")
        edit_scroll.pack(side="top", fill="both", expand=True, padx=0, pady=0)

        ctk.CTkLabel(edit_scroll, text="軌道編集", font=("", 14, "bold")).pack(pady=(5, 5))

        # グラデーション色
        color_sec = ctk.CTkFrame(edit_scroll)
        color_sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(color_sec, text="線の色").pack(anchor="w", padx=8, pady=(5, 0))

        start_row = ctk.CTkFrame(color_sec, fg_color="transparent")
        start_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(start_row, text="開始:", width=40).pack(side="left")
        self.edit_color_start_btn = ctk.CTkButton(
            start_row, text="", width=35, height=22,
            fg_color="#FFFF00", hover_color="#FFFF00",
            command=self._edit_pick_start_color
        )
        self.edit_color_start_btn.pack(side="left", padx=5)

        end_row = ctk.CTkFrame(color_sec, fg_color="transparent")
        end_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(end_row, text="終了:", width=40).pack(side="left")
        self.edit_color_end_btn = ctk.CTkButton(
            end_row, text="", width=35, height=22,
            fg_color="#FF0000", hover_color="#FF0000",
            command=self._edit_pick_end_color
        )
        self.edit_color_end_btn.pack(side="left", padx=5)

        # 太さ
        thick_sec = ctk.CTkFrame(edit_scroll)
        thick_sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(thick_sec, text="線の太さ").pack(anchor="w", padx=8, pady=(5, 0))
        self.edit_thick_label = ctk.CTkLabel(thick_sec, text="3 px")
        self.edit_thick_label.pack(anchor="e", padx=8)
        self.edit_thick_slider = ctk.CTkSlider(
            thick_sec, from_=1, to=10, number_of_steps=9,
            command=self._edit_on_thickness
        )
        self.edit_thick_slider.set(3)
        self.edit_thick_slider.pack(fill="x", padx=8, pady=(0, 8))

        # 軌跡終了マーカー
        end_sec = ctk.CTkFrame(edit_scroll)
        end_sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(end_sec, text="軌跡終了フレーム").pack(anchor="w", padx=8, pady=(5, 0))
        self.edit_end_frame_label = ctk.CTkLabel(end_sec, text="なし")
        self.edit_end_frame_label.pack(anchor="e", padx=8)
        end_btn_row = ctk.CTkFrame(end_sec, fg_color="transparent")
        end_btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkButton(end_btn_row, text="現フレームに設定", width=120,
                       command=self._edit_set_end_frame).pack(side="left", padx=2)
        ctk.CTkButton(end_btn_row, text="解除", width=60,
                       command=self._edit_clear_end_frame).pack(side="left", padx=2)

        # 不透明度
        alpha_sec = ctk.CTkFrame(edit_scroll)
        alpha_sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(alpha_sec, text="線の不透明度").pack(anchor="w", padx=8, pady=(5, 0))
        self.edit_alpha_label = ctk.CTkLabel(alpha_sec, text="85%")
        self.edit_alpha_label.pack(anchor="e", padx=8)
        self.edit_alpha_slider = ctk.CTkSlider(
            alpha_sec, from_=0, to=100, number_of_steps=100,
            command=self._edit_on_alpha
        )
        self.edit_alpha_slider.set(85)
        self.edit_alpha_slider.pack(fill="x", padx=8, pady=(0, 8))

        # エッジぼかし
        blur_sec = ctk.CTkFrame(edit_scroll)
        blur_sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(blur_sec, text="エッジぼかし").pack(anchor="w", padx=8, pady=(5, 0))
        self.edit_blur_label = ctk.CTkLabel(blur_sec, text="0")
        self.edit_blur_label.pack(anchor="e", padx=8)
        self.edit_blur_slider = ctk.CTkSlider(
            blur_sec, from_=0, to=20, number_of_steps=20,
            command=self._edit_on_blur
        )
        self.edit_blur_slider.set(0)
        self.edit_blur_slider.pack(fill="x", padx=8, pady=(0, 8))

        # フェードイン/アウト
        fade_sec = ctk.CTkFrame(edit_scroll)
        fade_sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(fade_sec, text="フェードイン/アウト").pack(anchor="w", padx=8, pady=(5, 0))
        self.edit_fade_label = ctk.CTkLabel(fade_sec, text="0 frames")
        self.edit_fade_label.pack(anchor="e", padx=8)
        self.edit_fade_slider = ctk.CTkSlider(
            fade_sec, from_=0, to=60, number_of_steps=60,
            command=self._edit_on_fade
        )
        self.edit_fade_slider.set(0)
        self.edit_fade_slider.pack(fill="x", padx=8, pady=(0, 8))

        # スイング情報
        self.edit_swing_label = ctk.CTkLabel(edit_scroll, text="Swing 1 (0 pts)")
        self.edit_swing_label.pack(padx=10, pady=5)

        # 編集状態
        self._edit_clip_id = None
        self._edit_cache = None
        self._edit_frame_no = 0
        self._edit_total = 0
        self._edit_swings = []      # [TrajectoryData, ...]
        self._edit_swing_idx = 0
        self._edit_playing = False
        self._edit_scale = 1.0
        self._edit_dragging = None

        # PanedWindow に左右を追加
        self.edit_paned.add(left, minsize=400, stretch="always")
        self.edit_paned.add(right, minsize=220, stretch="never")

        # 保存されたサッシ位置を復元
        self.edit_paned.bind("<Configure>", self._edit_paned_configure)
        self.edit_paned.bind("<ButtonRelease-1>", self._save_edit_sash)
        self._edit_sash_restored = False

        # 編集キャンバスのリサイズで再描画
        self.edit_canvas.bind("<Configure>", self._on_edit_canvas_resize)
        self._edit_canvas_resize_after = None

    def _edit_paned_configure(self, event):
        if self._edit_sash_restored:
            return
        saved_x = self.settings.data.get("edit_sash_x", 0)
        total_w = self.edit_paned.winfo_width()
        if saved_x > 0 and total_w > saved_x + 50:
            try:
                self.edit_paned.sash_place(0, saved_x, 0)
                self._edit_sash_restored = True
            except Exception:
                pass
        elif total_w > 500:
            # デフォルト: 右パネル280px
            try:
                self.edit_paned.sash_place(0, total_w - 280, 0)
                self._edit_sash_restored = True
            except Exception:
                pass

    def _save_edit_sash(self, event=None):
        try:
            coord = self.edit_paned.sash_coord(0)
            self.settings["edit_sash_x"] = coord[0]
            self.settings.save()
        except Exception:
            pass

    def _on_edit_canvas_resize(self, event=None):
        """編集キャンバスリサイズ時に再描画 (debounce)"""
        if self._edit_canvas_resize_after:
            try:
                self.after_cancel(self._edit_canvas_resize_after)
            except Exception:
                pass
        self._edit_canvas_resize_after = self.after(
            100, self._edit_redraw_after_resize)

    def _edit_redraw_after_resize(self):
        self._edit_canvas_resize_after = None
        if self._edit_clip_id:
            try:
                self._edit_update_display()
            except Exception:
                pass

    def _refresh_edit_clips_list(self):
        """編集タブのクリップリストを更新"""
        for w in self._edit_clip_widgets:
            w.destroy()
        self._edit_clip_widgets.clear()

        for i, clip in enumerate(self.clip_manager.clips):
            row = ctk.CTkFrame(self.edit_clips_scroll, height=30)
            row.grid(row=i, column=0, sticky="ew", pady=1)
            row.grid_columnconfigure(0, weight=1)

            # 選択中のクリップはハイライト
            is_current = (self._edit_clip_id == clip.id)
            fg = "#1a3a1a" if is_current else "transparent"
            border_w = 1 if is_current else 0
            row.configure(fg_color=fg, border_width=border_w)
            if is_current:
                row.configure(border_color="#00AA00")

            name_btn = ctk.CTkButton(
                row, text=clip.name, anchor="w",
                font=("", 12),
                fg_color="transparent", hover_color="#333333",
                command=lambda cid=clip.id: self._edit_select_clip(cid)
            )
            name_btn.grid(row=0, column=0, sticky="ew", padx=3)

            dur = f"{clip.duration_sec:.1f}s"
            traj = "✓" if clip.has_trajectory else ""
            ctk.CTkLabel(row, text=f"{dur} {traj}", width=70,
                         font=("", 11)).grid(row=0, column=1, padx=3)

            self._edit_clip_widgets.append(row)

    def _edit_select_clip(self, clip_id):
        """編集タブでクリップを選択して軌道編集にロード"""
        # 現在編集中の軌道を先に自動保存
        if self._edit_clip_id and self._edit_clip_id != clip_id:
            self._edit_autosave_trajectory()
        self._edit_clip_id = clip_id
        self._load_edit_clip()
        self._refresh_edit_clips_list()

    @property
    def _edit_current_swing(self):
        if self._edit_swings and 0 <= self._edit_swing_idx < len(self._edit_swings):
            return self._edit_swings[self._edit_swing_idx]
        return None

    def _load_edit_clip(self):
        """編集タブにクリップをロード (最初のフレームを即表示、残りはバックグラウンド)"""
        clip = self.clip_manager.get_clip(self._edit_clip_id)
        if not clip:
            return

        print(f"[Edit] クリップ読み込み: {clip.name}")
        self._edit_direct_clip = None
        self._edit_cache = FrameCache(clip.source_path, clip.in_frame, clip.get_out_frame())
        self._edit_total = len(self._edit_cache)
        self._edit_frame_no = 0

        # 既存軌道を読み込み
        saved = self.clip_manager.load_trajectory(self._edit_clip_id)
        if saved:
            self._edit_swings = saved
        else:
            preset = GRADIENT_PRESETS[0]
            self._edit_swings = [TrajectoryData(
                color_start_hex=preset[0], color_end_hex=preset[1], thickness=3
            )]
        self._edit_swing_idx = 0

        self.edit_slider.configure(to=max(self._edit_total - 1, 1))
        self._edit_update_color_buttons()
        self._edit_update_display()

        # 残りフレームをバックグラウンドで読み込み
        self._edit_cache.load_remaining()

    def _edit_update_color_buttons(self):
        swing = self._edit_current_swing
        if swing:
            self.edit_color_start_btn.configure(
                fg_color=swing.color_start_hex, hover_color=swing.color_start_hex)
            self.edit_color_end_btn.configure(
                fg_color=swing.color_end_hex, hover_color=swing.color_end_hex)
            self.edit_thick_slider.set(swing.thickness)
            self.edit_thick_label.configure(text=f"{swing.thickness} px")
            blur = getattr(swing, "blur", 0)
            self.edit_blur_slider.set(blur)
            self.edit_blur_label.configure(text=f"{blur}")
            fade = getattr(swing, "fade_frames", 0)
            self.edit_fade_slider.set(fade)
            self.edit_fade_label.configure(text=f"{fade} frames")
            alpha = getattr(swing, "alpha", 0.85)
            self.edit_alpha_slider.set(int(alpha * 100))
            self.edit_alpha_label.configure(text=f"{int(alpha * 100)}%")
            self._edit_update_end_frame_label()

    def _edit_update_display(self):
        if not self._edit_cache or self._edit_total == 0:
            return

        frame = self._edit_cache[self._edit_frame_no]
        if frame is None:
            return

        # 軌道描画 (現在フレームまでアニメーション)
        cur = self._edit_frame_no
        for swing in self._edit_swings:
            if len(swing.points) < 2:
                # マーカーのみ (点数不足)
                if swing.points:
                    draw_markers(frame, [p for p in swing.points if p[2] <= cur],
                                 hex_to_bgr(swing.color_start_hex),
                                 hex_to_bgr(swing.color_end_hex), MARKER_RADIUS)
                continue

            end_f = getattr(swing, 'end_frame', -1)
            blur = getattr(swing, 'blur', 0)
            base_a = getattr(swing, 'alpha', 0.85)
            eff_alpha = _compute_fade_alpha(swing, cur, base_alpha=base_a)
            hide_trail = eff_alpha <= 0.0

            if not hide_trail:
                ts = TimedSpline(swing.points, SPLINE_RESOLUTION)
                curve_pts = ts.get_curve_at_frame(cur)
                if curve_pts and len(curve_pts) >= 2:
                    c_start = hex_to_bgr(swing.color_start_hex)
                    c_end = hex_to_bgr(swing.color_end_hex)
                    full_len = len(ts._curve)
                    ratio = len(curve_pts) / max(full_len, 1)
                    c_end_anim = lerp_color_bgr(c_start, c_end, ratio)
                    draw_gradient_trail(frame, curve_pts, c_start,
                                        c_end_anim, swing.thickness,
                                        eff_alpha, blur=blur)
            # マーカー (現在フレーム以前のみ表示)
            if swing.points and not hide_trail:
                visible_pts = [p for p in swing.points if p[2] <= cur]
                if visible_pts:
                    draw_markers(frame, visible_pts,
                                 hex_to_bgr(swing.color_start_hex),
                                 hex_to_bgr(swing.color_end_hex), MARKER_RADIUS)

        cw = self.edit_canvas.winfo_width()
        ch = self.edit_canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 450

        self._edit_photo, self._edit_scale = frame_to_photo(frame, cw, ch)
        self.edit_canvas.delete("all")
        self.edit_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                       image=self._edit_photo)

        self.edit_frame_label.configure(text=f"{self._edit_frame_no} / {self._edit_total - 1}")
        self.edit_slider.set(self._edit_frame_no)

        swing = self._edit_current_swing
        pts = len(swing.points) if swing else 0
        self.edit_swing_label.configure(
            text=f"Swing {self._edit_swing_idx + 1} ({pts} pts)")

    def _edit_canvas_to_video(self, cx, cy):
        cw = self.edit_canvas.winfo_width()
        ch = self.edit_canvas.winfo_height()
        if not self._edit_cache or self._edit_total == 0:
            return 0, 0

        frame = self._edit_cache[0]
        if frame is None:
            return 0, 0
        fh, fw = frame.shape[:2]

        scale = self._edit_scale
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        ox = (cw - disp_w) / 2
        oy = (ch - disp_h) / 2

        vx = int((cx - ox) / scale)
        vy = int((cy - oy) / scale)
        return vx, vy

    def _edit_left_click(self, event):
        swing = self._edit_current_swing
        if not swing:
            return
        vx, vy = self._edit_canvas_to_video(event.x, event.y)
        swing.points.append((vx, vy, self._edit_frame_no))
        swing.points.sort(key=lambda p: p[2])
        print(f"  Point: ({vx}, {vy}) @ frame {self._edit_frame_no}")
        self._edit_update_display()

    def _edit_find_nearest(self, vx, vy):
        best = None
        for si, swing in enumerate(self._edit_swings):
            for pi, pt in enumerate(swing.points):
                d = np.hypot(vx - pt[0], vy - pt[1])
                if best is None or d < best[2]:
                    best = (si, pi, d)
        return best

    def _edit_right_press(self, event):
        vx, vy = self._edit_canvas_to_video(event.x, event.y)
        result = self._edit_find_nearest(vx, vy)
        thresh = POINT_GRAB_RADIUS / max(self._edit_scale, 0.1)
        if result and result[2] < thresh:
            self._edit_dragging = (result[0], result[1])
            self.edit_canvas.config(cursor="fleur")
        else:
            self._edit_dragging = None

    def _edit_right_drag(self, event):
        if self._edit_dragging is None:
            return
        si, pi = self._edit_dragging
        vx, vy = self._edit_canvas_to_video(event.x, event.y)
        old = self._edit_swings[si].points[pi]
        self._edit_swings[si].points[pi] = (vx, vy, old[2])
        self._edit_update_display()

    def _edit_right_release(self, event):
        self._edit_dragging = None
        self.edit_canvas.config(cursor="")
        self._edit_update_display()

    def _edit_jump(self, delta):
        self._edit_frame_no = max(0, min(self._edit_frame_no + delta, self._edit_total - 1))
        self._edit_update_display()

    def _on_frame_step_change(self, value):
        """フレーム送りステップ変更"""
        try:
            self._frame_step = int(value)
        except (ValueError, TypeError):
            self._frame_step = 1
        # 送出タブの表示も同期
        if hasattr(self, "po_step_seg"):
            self.po_step_seg.set(str(self._frame_step))
        if hasattr(self, "edit_step_seg"):
            self.edit_step_seg.set(str(self._frame_step))

    def _on_edit_slider(self, value):
        self._edit_frame_no = int(value)
        self._edit_update_display()

    def _edit_toggle_play(self):
        self._edit_playing = not self._edit_playing
        self.edit_play_btn.configure(text="⏸" if self._edit_playing else "▶")
        if self._edit_playing:
            self._edit_play_loop()

    def _edit_play_loop(self):
        if not self._edit_playing:
            return
        if self._edit_frame_no >= self._edit_total - 1:
            self._edit_playing = False
            self.edit_play_btn.configure(text="▶")
            return
        self._edit_frame_no += 1
        self._edit_update_display()
        fps = 29.97
        clip = self.clip_manager.get_clip(self._edit_clip_id) if self._edit_clip_id else None
        if clip:
            fps = clip.fps
        self.after(int(1000 / fps), self._edit_play_loop)

    def _edit_pick_start_color(self):
        swing = self._edit_current_swing
        if not swing:
            return
        color = colorchooser.askcolor(initialcolor=swing.color_start_hex, title="開始色")
        if color[1]:
            swing.color_start_hex = color[1]
            self._edit_update_color_buttons()
            self._edit_update_display()

    def _edit_pick_end_color(self):
        swing = self._edit_current_swing
        if not swing:
            return
        color = colorchooser.askcolor(initialcolor=swing.color_end_hex, title="終了色")
        if color[1]:
            swing.color_end_hex = color[1]
            self._edit_update_color_buttons()
            self._edit_update_display()

    def _edit_on_thickness(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.thickness = int(value)
            self.edit_thick_label.configure(text=f"{swing.thickness} px")
            self._edit_update_display()

    def _edit_on_blur(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.blur = int(value)
            self.edit_blur_label.configure(text=f"{swing.blur}")
            self._edit_update_display()

    def _edit_on_fade(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.fade_frames = int(value)
            self.edit_fade_label.configure(text=f"{swing.fade_frames} frames")
            self._edit_update_display()

    def _edit_on_alpha(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.alpha = max(0.0, min(1.0, int(value) / 100.0))
            self.edit_alpha_label.configure(text=f"{int(swing.alpha * 100)}%")
            self._edit_update_display()

    def _edit_set_end_frame(self):
        """軌跡終了フレームを現在フレームに設定"""
        swing = self._edit_current_swing
        if swing:
            swing.end_frame = self._edit_frame_no
            self._edit_update_end_frame_label()
            self._edit_update_display()

    def _edit_clear_end_frame(self):
        """軌跡終了フレームを解除"""
        swing = self._edit_current_swing
        if swing:
            swing.end_frame = -1
            self._edit_update_end_frame_label()
            self._edit_update_display()

    def _edit_update_end_frame_label(self):
        swing = self._edit_current_swing
        if swing and swing.end_frame >= 0:
            self.edit_end_frame_label.configure(text=f"f{swing.end_frame}")
        else:
            self.edit_end_frame_label.configure(text="なし")

    def _edit_next_swing(self):
        idx = len(self._edit_swings)
        preset = GRADIENT_PRESETS[idx % len(GRADIENT_PRESETS)]
        t = self._edit_current_swing.thickness if self._edit_current_swing else 3
        self._edit_swings.append(TrajectoryData(
            color_start_hex=preset[0], color_end_hex=preset[1], thickness=t))
        self._edit_swing_idx = idx
        self._edit_update_color_buttons()
        self._edit_update_display()

    def _edit_clear_swing(self):
        swing = self._edit_current_swing
        if swing:
            swing.points.clear()
            self._edit_update_display()

    def _edit_undo(self):
        swing = self._edit_current_swing
        if swing and swing.points:
            swing.points.pop()
            self._edit_update_display()

    def _edit_delete_trajectory(self):
        """軌道を削除 (メモリ上の全スイング + 保存ファイル)"""
        if not self._edit_clip_id:
            return
        clip = self.clip_manager.get_clip(self._edit_clip_id)

        # --- 保存ファイルを削除 ---
        if clip and clip.trajectory_path:
            try:
                tp = Path(clip.trajectory_path)
                if tp.exists():
                    tp.unlink()
                    print(f"[Edit] 軌道ファイル削除: {tp.name}")
            except Exception as e:
                print(f"[Edit] 軌道ファイル削除エラー: {e}")

        # --- ClipData 更新 ---
        if clip:
            clip.trajectory_path = ""
            clip.has_trajectory = False
            self.clip_manager.save()

        # --- 送出リスト内の同一クリップのスイングもクリア ---
        playlist_updated = False
        for item in self.playout.playlist:
            if clip and item.clip.id == clip.id and item.swings:
                item.swings = []
                playlist_updated = True
        if playlist_updated:
            self.playout.save_playlist(self._playout_json)
            self._refresh_playout_list()

        # --- メモリ上の編集スイングをリセット ---
        preset = GRADIENT_PRESETS[0]
        self._edit_swings = [TrajectoryData(
            color_start_hex=preset[0], color_end_hex=preset[1], thickness=3)]
        self._edit_swing_idx = 0

        self._edit_update_color_buttons()
        self._edit_update_display()
        self._refresh_clips_list()
        self._refresh_edit_clips_list()
        print(f"[Edit] 軌道を削除: {clip.name if clip else ''}")

    def _edit_save_trajectory(self):
        if not self._edit_clip_id:
            return
        self.clip_manager.save_trajectory(self._edit_clip_id, self._edit_swings)
        self._refresh_clips_list()
        print("[Edit] 軌道を保存しました")

    def _edit_autosave_trajectory(self):
        """軌道を静かに自動保存 (点が1つ以上ある場合のみ)"""
        if not self._edit_clip_id:
            return
        has_any = any(s.points for s in self._edit_swings)
        if not has_any:
            return
        try:
            self.clip_manager.save_trajectory(self._edit_clip_id, self._edit_swings)
        except Exception as e:
            print(f"[Edit] 自動保存エラー: {e}")

    def _edit_export_video(self):
        """軌道付き動画を書き出し"""
        if not self._edit_cache:
            return

        clip = None
        if self._edit_clip_id:
            clip = self.clip_manager.get_clip(self._edit_clip_id)
        if not clip:
            clip = getattr(self, '_edit_direct_clip', None)
        if not clip:
            return

        # 書き出し前に現在の軌道を自動保存
        self._edit_autosave_trajectory()

        # スナップショットを取得 (バックグラウンド中にユーザーが別クリップを開いても安全)
        cache = self._edit_cache
        total = self._edit_total
        swings_copy = []
        for s in self._edit_swings:
            swings_copy.append(TrajectoryData(
                points=list(s.points),
                color_start_hex=s.color_start_hex,
                color_end_hex=s.color_end_hex,
                thickness=s.thickness,
                end_frame=getattr(s, "end_frame", -1),
                blur=getattr(s, "blur", 0),
                fade_frames=getattr(s, "fade_frames", 0),
            ))

        out_path = self.project_dir / "exports" / f"swing_{clip.name}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.edit_frame_label.configure(text="書き出し中...")

        thread = threading.Thread(
            target=self._do_edit_export,
            args=(clip, swings_copy, out_path, cache, total),
            daemon=True
        )
        thread.start()

    def _do_edit_export(self, clip, swings, out_path, cache, total):
        print(f"[Edit] 動画書き出し中... ({total} frames)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, clip.fps,
                                  (clip.width, clip.height))

        # スプライン事前構築
        spline_data = []
        for swing in swings:
            if len(swing.points) < 2:
                continue
            spline_data.append({
                "spline": TimedSpline(swing.points, SPLINE_RESOLUTION),
                "color_start": hex_to_bgr(swing.color_start_hex),
                "color_end": hex_to_bgr(swing.color_end_hex),
                "thickness": swing.thickness,
                "end_frame": getattr(swing, "end_frame", -1),
                "blur": getattr(swing, "blur", 0),
                "swing_ref": swing,
            })

        # 元ファイルから直接読み込み (キャッシュが未完了でも全フレーム書き出し)
        src_path = clip.exported_path if clip.exported_path else clip.source_path
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[Edit] ソースを開けません: {src_path}")
            writer.release()
            self.after(0, lambda: self.edit_frame_label.configure(
                text="書き出し失敗: ソースを開けません"))
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.in_frame)
        written = 0
        for i in range(total):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            for sd in spline_data:
                base_a = getattr(sd["swing_ref"], "alpha", 0.85)
                eff_alpha = _compute_fade_alpha(sd["swing_ref"], i, base_alpha=base_a)
                if eff_alpha <= 0.0:
                    continue
                curve_pts = sd["spline"].get_curve_at_frame(i)
                if curve_pts and len(curve_pts) >= 2:
                    full_len = len(sd["spline"]._curve)
                    ratio = len(curve_pts) / max(full_len, 1)
                    c_end = lerp_color_bgr(sd["color_start"], sd["color_end"], ratio)
                    draw_gradient_trail(frame, curve_pts, sd["color_start"],
                                        c_end, sd["thickness"],
                                        eff_alpha, blur=sd["blur"])

            writer.write(frame)
            written += 1
            if (i + 1) % 100 == 0:
                pct = int((i + 1) / total * 100)
                self.after(0, lambda p=pct: self.edit_frame_label.configure(
                    text=f"書き出し中... {p}%"))

        cap.release()
        writer.release()
        # 書き出し完了通知のみ (送出リストには追加しない)
        self.after(0, lambda p=out_path: self.edit_frame_label.configure(
            text=f"書き出し完了: {p.name}"))
        print(f"[Edit] 出力: {out_path} ({written}/{total} frames)")

    def _add_export_to_playout(self, export_path):
        """書き出した動画を送出リストに自動追加 (クリップには追加しない)"""
        try:
            p = Path(export_path)
            cap = cv2.VideoCapture(str(p))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            import datetime
            clip_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            duration = total / max(fps, 1)
            clip = ClipData(
                id=clip_id,
                source_path=str(p),
                name=p.stem,
                width=w, height=h,
                fps=fps,
                total_frames=total,
                in_frame=0,
                out_frame=total - 1,
                duration_sec=duration,
            )
            self.playout.add_item(clip, [])
            self.playout.save_playlist(self._playout_json)
            self._refresh_playout_list()
            self.edit_frame_label.configure(
                text=f"書き出し完了: {p.name}\n"
                     f"→ 送出リストに追加しました")
            print(f"[Playout] 追加: {clip.name}")
        except Exception as e:
            self.edit_frame_label.configure(
                text=f"書き出し完了 (送出追加エラー: {e})")
            print(f"[Edit] 送出追加エラー: {e}")

    # =========================================================================
    # 送出タブ
    # =========================================================================
    def _build_playout_tab(self):
        tab = self.tab_playout
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # PanedWindow でリサイズ可能な左右分割
        self.po_paned = PanedWindow(
            tab, orient="horizontal", sashwidth=6,
            bg="#2b2b2b", sashrelief="flat", borderwidth=0,
        )
        self.po_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 左: プレビュー
        left = ctk.CTkFrame(self.po_paned, fg_color="transparent")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        self.playout_canvas = Canvas(left, bg="black", highlightthickness=0)
        self.playout_canvas.grid(row=0, column=0, sticky="nsew")
        self._playout_photo = None
        self._po_pending_frame = None   # 最新フレーム (スレッドから書き込み)
        self._po_gui_scheduled = False  # after() 登録済みフラグ
        self._po_canvas_size = (800, 450)  # キャンバスサイズキャッシュ

        # シークバー + フレーム表示
        seek_frame_w = ctk.CTkFrame(left, fg_color="transparent")
        seek_frame_w.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        seek_frame_w.grid_columnconfigure(1, weight=1)

        self.po_frame_label = ctk.CTkLabel(seek_frame_w, text="0 / 0", width=120)
        self.po_frame_label.grid(row=0, column=0, padx=(0, 10))

        self.po_seek_slider = ctk.CTkSlider(seek_frame_w, from_=0, to=100,
                                              command=self._on_playout_seek)
        self.po_seek_slider.grid(row=0, column=1, sticky="ew")
        self.po_seek_slider.set(0)
        self._po_slider_updating = False  # 再生中の自動更新フラグ

        # 送出コントロール (プロ用)
        po_ctrl = ctk.CTkFrame(left)
        po_ctrl.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        # PLAY / PAUSE トグルボタン
        self.po_play_btn = ctk.CTkButton(
            po_ctrl, text="▶ PLAY", width=140, height=55,
            font=("", 20, "bold"),
            fg_color="#006400", hover_color="#228B22",
            command=self._playout_toggle_play
        )
        self.po_play_btn.pack(side="left", padx=5, pady=5)

        # CUE (頭出し)
        ctk.CTkButton(po_ctrl, text="CUE", width=80, height=55,
                       font=("", 16, "bold"),
                       fg_color="#B8860B", hover_color="#DAA520",
                       command=self._playout_cue_top).pack(side="left", padx=3)

        # PREV / NEXT
        ctk.CTkButton(po_ctrl, text="⏮ PREV", width=80, height=55,
                       font=("", 13, "bold"),
                       fg_color="#333", hover_color="#555",
                       command=self._playout_prev).pack(side="left", padx=3)
        ctk.CTkButton(po_ctrl, text="NEXT ⏭", width=80, height=55,
                       font=("", 13, "bold"),
                       fg_color="#333", hover_color="#555",
                       command=self._playout_next).pack(side="left", padx=3)

        # スロー再生速度セレクタ
        speed_frame = ctk.CTkFrame(po_ctrl, fg_color="transparent")
        speed_frame.pack(side="left", padx=10)
        ctk.CTkLabel(speed_frame, text="速度", font=("", 11)).pack()
        self.po_speed_seg = ctk.CTkSegmentedButton(
            speed_frame, values=["1x", "1/2", "1/4", "1/8"],
            font=("", 13, "bold"), width=200,
            command=self._on_speed_change)
        self.po_speed_seg.pack()
        self.po_speed_seg.set("1x")
        self._po_speed = 1.0

        # フレーム送りステップ
        step_frame = ctk.CTkFrame(po_ctrl, fg_color="transparent")
        step_frame.pack(side="left", padx=10)
        ctk.CTkLabel(step_frame, text="Step", font=("", 11)).pack()
        self.po_step_seg = ctk.CTkSegmentedButton(
            step_frame, values=["1", "2", "5", "10"],
            font=("", 13, "bold"), width=160,
            command=self._on_frame_step_change)
        self.po_step_seg.pack()
        self.po_step_seg.set(str(self._frame_step))

        self.po_status = ctk.CTkLabel(po_ctrl, text="STOPPED", font=("", 14, "bold"))
        self.po_status.pack(side="left", padx=10)

        # 右: プレイリスト
        right = ctk.CTkFrame(self.po_paned, width=350)

        ctk.CTkLabel(right, text="送出リスト", font=("", 16, "bold")).pack(pady=10)

        self.playout_scroll = ctk.CTkScrollableFrame(right)
        self.playout_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.playout_scroll.grid_columnconfigure(0, weight=1)
        self._playout_widgets = []

        # ShuttlePRO v2 ステータス
        shuttle_frame = ctk.CTkFrame(right)
        shuttle_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(shuttle_frame, text="ShuttlePRO v2",
                      font=("", 12, "bold")).pack(anchor="w", padx=8, pady=(5, 2))
        self.po_shuttle_status = ctk.CTkLabel(
            shuttle_frame, text="未接続", font=("", 11),
            text_color="#888")
        self.po_shuttle_status.pack(anchor="w", padx=8)
        self.po_shuttle_info = ctk.CTkLabel(
            shuttle_frame, text="", font=("", 11),
            text_color="#00CCFF")
        self.po_shuttle_info.pack(anchor="w", padx=8, pady=(0, 5))

        # ショートカットヘルプ
        help_frame = ctk.CTkFrame(right)
        help_frame.pack(fill="x", padx=5, pady=5)
        help_text = (
            "KB: Space=PLAY/PAUSE  Esc=CUE\n"
            "Enter=PLAY  1-4=速度\n"
            "A/D=1F  W/S=5F  N/P=NEXT/PREV\n"
            "\nShuttlePRO:\n"
            "ジョグ=フレーム送り\n"
            "シャトル=可変速再生\n"
            "Btn 1:PREV 2:CUE 3:PLAY\n"
            "    4:NEXT 5-8:速度 9:STOP"
        )
        ctk.CTkLabel(help_frame, text=help_text, font=("", 10),
                      justify="left", text_color="gray").pack(padx=8, pady=5)

        self._playout_selected_idx = None

        # PanedWindow に左右を追加 (最小幅を指定)
        self.po_paned.add(left, minsize=400, stretch="always")
        self.po_paned.add(right, minsize=200, stretch="never")

        # 保存されたサッシ位置を復元 (タブ表示後)
        self.po_paned.bind("<Configure>", self._playout_paned_configure)
        self.po_paned.bind("<ButtonRelease-1>", self._save_playout_sash)
        self._playout_sash_restored = False

        # 送出キャンバスのリサイズで再描画
        self.playout_canvas.bind("<Configure>", self._on_playout_canvas_resize)
        self._playout_canvas_resize_after = None

    def _on_playout_canvas_resize(self, event=None):
        """送出キャンバスリサイズ時に再描画 (debounce)"""
        if self._playout_canvas_resize_after:
            try:
                self.after_cancel(self._playout_canvas_resize_after)
            except Exception:
                pass
        self._playout_canvas_resize_after = self.after(
            100, self._playout_redraw_after_resize)

    def _playout_redraw_after_resize(self):
        self._playout_canvas_resize_after = None
        # 再生中は次フレームで自動更新されるので停止中のみ再描画
        try:
            if not self.playout._playing:
                self._playout_show_preview()
        except Exception:
            pass

    def _playout_paned_configure(self, event):
        """PanedWindow初回表示時にサッシ位置を復元"""
        if self._playout_sash_restored:
            return
        saved_x = self.settings.data.get("playout_sash_x", 0)
        if saved_x > 0:
            total_w = self.po_paned.winfo_width()
            if total_w > saved_x + 50:
                try:
                    self.po_paned.sash_place(0, saved_x, 0)
                    self._playout_sash_restored = True
                except Exception:
                    pass
        elif self.po_paned.winfo_width() > 400:
            # 未保存時はデフォルト: 右パネルを350pxに
            try:
                total_w = self.po_paned.winfo_width()
                self.po_paned.sash_place(0, total_w - 350, 0)
                self._playout_sash_restored = True
            except Exception:
                pass

    def _save_playout_sash(self, event=None):
        """サッシ位置を設定に保存"""
        try:
            coord = self.po_paned.sash_coord(0)
            self.settings["playout_sash_x"] = coord[0]
            self.settings.save()
        except Exception:
            pass

    def _refresh_playout_list(self):
        for w in self._playout_widgets:
            w.destroy()
        self._playout_widgets.clear()

        for i, item in enumerate(self.playout.playlist):
            row = ctk.CTkFrame(self.playout_scroll, height=35)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            # 選択ボタン (キューアップ)
            ctk.CTkButton(
                row, text="選択", width=50, height=28,
                font=("", 12, "bold"),
                fg_color="#B8860B", hover_color="#DAA520",
                command=lambda idx=i: self._playout_select(idx)
            ).grid(row=0, column=0, padx=3)

            # クリップ名 (swing_ プレフィックスは省略)
            display_name = item.clip.name
            if display_name.startswith("swing_"):
                display_name = display_name[len("swing_"):]
            ctk.CTkLabel(row, text=display_name, anchor="w",
                         font=("", 12)).grid(row=0, column=1, sticky="ew", padx=3)

            dur = f"{item.clip.duration_sec:.1f}s"
            ctk.CTkLabel(row, text=dur, width=60).grid(row=0, column=2, padx=3)

            # 削除ボタン
            ctk.CTkButton(
                row, text="×", width=28, height=28,
                fg_color="#8B0000", hover_color="#A52A2A",
                font=("", 14, "bold"),
                command=lambda idx=i: self._playout_remove_item(idx),
            ).grid(row=0, column=3, padx=(0, 3))

            self._playout_widgets.append(row)

    def _on_playout_seek(self, value):
        """シークバー操作 (スロットル付き)"""
        if self._po_slider_updating:
            return
        if self.playout._playing and not self.playout._paused:
            return  # 再生中はユーザーシーク無効 (一時停止中は許可)
        # スロットル: 最後のリクエストだけ実行 (50ms後)
        self._po_seek_pending = int(value)
        if not getattr(self, '_po_seek_timer', None):
            self._po_seek_timer = self.after(50, self._do_playout_seek)

    def _do_playout_seek(self):
        """スロットル済みシーク実行"""
        self._po_seek_timer = None
        if self.playout._playing and not self.playout._paused:
            return
        frame_offset = getattr(self, '_po_seek_pending', 0)
        self.playout.seek_to(frame_offset)
        self._playout_show_preview(frame_offset)

    def _playout_seek_delta(self, delta):
        """送出タブでフレーム送り/戻り (停止中 or 一時停止中)"""
        if self.playout._playing and not self.playout._paused:
            return
        total = self.playout.get_cued_total_frames()
        if total <= 0:
            return
        current = int(self.po_seek_slider.get())
        new_pos = max(0, min(current + delta, total - 1))
        self._po_slider_updating = True
        self.po_seek_slider.set(new_pos)
        self._po_slider_updating = False
        self.playout.seek_to(new_pos)
        self._playout_show_preview(new_pos)

    def _playout_show_preview(self, frame_offset=None):
        """送出プレビューを即時更新 (シーク/キュー後)"""
        pf = self.playout.preview_frame
        if pf is None:
            return
        cw = self.playout_canvas.winfo_width()
        ch = self.playout_canvas.winfo_height()
        if cw > 10 and ch > 10:
            self._playout_photo, _ = frame_to_photo(pf, cw, ch)
            self.playout_canvas.delete("all")
            self.playout_canvas.create_image(
                cw // 2, ch // 2, anchor="center", image=self._playout_photo)
        if frame_offset is not None:
            total = self.playout.get_cued_total_frames()
            self.po_frame_label.configure(text=f"{frame_offset} / {total - 1}")

    def _playout_select(self, idx):
        self._playout_selected_idx = idx
        self.playout.cue(idx)  # cue() が内部で停止+キューを安全に行う
        # シークバー更新
        total = self.playout.get_cued_total_frames()
        self._po_slider_updating = True
        self.po_seek_slider.configure(to=max(total - 1, 1))
        self.po_seek_slider.set(0)
        self._po_slider_updating = False
        self.po_status.configure(text="⏹ CUED", text_color="#FFFF00")
        self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                    hover_color="#228B22")
        # ハイライト更新
        self._playout_highlight_row(idx)
        # キュー時のプレビュー即表示
        self._playout_show_preview(0)

    def _playout_highlight_row(self, idx):
        """プレイリストの選択行をハイライト"""
        for i, w in enumerate(self._playout_widgets):
            if i == idx:
                w.configure(fg_color="#1a3a1a", border_color="#00AA00", border_width=1)
            else:
                w.configure(fg_color=("gray86", "gray17"), border_width=0)

    def _playout_play(self):
        if not self.playout.playlist:
            return

        # DeckLink出力開始
        if not self.deck_output:
            self.deck_output = DeckLinkOutput(
                self.settings["output_device"],
                self.settings["width"], self.settings["height"],
                self.settings["fps"],
            )
            self.deck_output.start()

        self.playout.output_device = self.deck_output
        self.playout.on_frame_update = self._on_playout_frame
        self.playout.on_clip_changed = self._on_playout_clip_changed
        self.playout.on_playback_ended = self._on_playout_ended
        self.playout.play()
        self._update_play_status()

    def _update_play_status(self):
        """再生中の速度表示を更新"""
        spd = self._po_speed
        if spd < 1.0:
            labels = {0.5: "1/2", 0.25: "1/4", 0.125: "1/8"}
            s = labels.get(spd, f"{spd:.2f}")
            self.po_status.configure(text=f"▶ SLOW {s}", text_color="#00CCFF")
        else:
            self.po_status.configure(text="▶ PLAYING", text_color="#00FF00")

    def _playout_stop(self):
        self.playout.stop()  # 非ブロッキング
        self.po_status.configure(text="⏹ STOPPED", text_color="white")
        self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                    hover_color="#228B22")

    def _playout_toggle_play(self):
        """PLAY / PAUSE トグル"""
        if self.playout._playing and not self.playout._paused:
            # 再生中 → 一時停止
            self.playout.pause()
            self.po_status.configure(text="⏸ PAUSED", text_color="#FFAA00")
            self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                        hover_color="#228B22")
        elif self.playout._paused:
            # 一時停止中 → 再開
            self.playout.play()
            self._update_play_status()
            self.po_play_btn.configure(text="⏸ PAUSE", fg_color="#B8860B",
                                        hover_color="#DAA520")
        else:
            # 停止中 → 再生開始
            self._playout_play()
            self.po_play_btn.configure(text="⏸ PAUSE", fg_color="#B8860B",
                                        hover_color="#DAA520")

    def _on_speed_change(self, value):
        """スロー再生速度変更"""
        speed_map = {"1x": 1.0, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125}
        self._po_speed = speed_map.get(value, 1.0)
        self.playout.speed = self._po_speed
        if self.playout._playing and not self.playout._paused:
            self._update_play_status()

    def _playout_cue_top(self):
        """CUE: 現在クリップの先頭に戻す"""
        was_playing = self.playout._playing
        if was_playing:
            self.playout.stop()
        idx = self.playout.current_index
        if 0 <= idx < len(self.playout.playlist):
            self.playout.cue(idx)
            self._po_slider_updating = True
            total = self.playout.get_cued_total_frames()
            self.po_seek_slider.configure(to=max(total - 1, 1))
            self.po_seek_slider.set(0)
            self.po_frame_label.configure(text=f"0 / {total - 1}")
            self._po_slider_updating = False
            self.po_status.configure(text="⏹ CUED", text_color="#FFFF00")
            self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                        hover_color="#228B22")

    def _playout_next(self):
        """次のクリップ (停止→cue→再生)"""
        was_playing = self.playout._playing
        if was_playing:
            self.playout.stop()
        if self.playout.current_index < len(self.playout.playlist) - 1:
            idx = self.playout.current_index + 1
            self.playout.cue(idx)  # 安全にcue
            self._playout_selected_idx = idx
            total = self.playout.get_cued_total_frames()
            self._po_slider_updating = True
            self.po_seek_slider.configure(to=max(total - 1, 1))
            self.po_seek_slider.set(0)
            self.po_frame_label.configure(text=f"0 / {total - 1}")
            self._po_slider_updating = False
            self._playout_highlight_row(idx)
            if was_playing:
                self._playout_play()

    def _playout_prev(self):
        """前のクリップ (停止→cue→再生)"""
        was_playing = self.playout._playing
        if was_playing:
            self.playout.stop()
        if self.playout.current_index > 0:
            idx = self.playout.current_index - 1
            self.playout.cue(idx)  # 安全にcue
            self._playout_selected_idx = idx
            total = self.playout.get_cued_total_frames()
            self._po_slider_updating = True
            self.po_seek_slider.configure(to=max(total - 1, 1))
            self.po_seek_slider.set(0)
            self.po_frame_label.configure(text=f"0 / {total - 1}")
            self._po_slider_updating = False
            self._playout_highlight_row(idx)
            if was_playing:
                self._playout_play()

    def _playout_remove_selected(self):
        if self._playout_selected_idx is not None:
            self._playout_remove_item(self._playout_selected_idx)

    def _playout_remove_item(self, idx):
        """送出リストの行ボタンから直接削除 (実ファイルも削除)"""
        if 0 <= idx < len(self.playout.playlist):
            item = self.playout.playlist[idx]
            # 実ファイル削除
            src = Path(item.clip.source_path)
            if src.exists():
                try:
                    src.unlink()
                    print(f"[Playout] ファイル削除: {src.name}")
                except Exception as e:
                    print(f"[Playout] ファイル削除エラー: {src.name}: {e}")
        self.playout.remove_item(idx)
        if self._playout_selected_idx == idx:
            self._playout_selected_idx = None
        elif self._playout_selected_idx is not None and self._playout_selected_idx > idx:
            self._playout_selected_idx -= 1
        self.playout.save_playlist(self._playout_json)
        self._refresh_playout_list()

    def _playout_clear(self):
        self.playout.playlist.clear()
        self._refresh_playout_list()

    def _playout_open_in_edit(self, idx):
        """送出リストのアイテムを編集タブで開く"""
        if idx < 0 or idx >= len(self.playout.playlist):
            return
        item = self.playout.playlist[idx]
        clip = item.clip

        print(f"[Edit] 送出クリップ読み込み: {clip.name}")
        self._edit_clip_id = None
        self._edit_direct_clip = clip
        self._edit_cache = FrameCache(clip.source_path, clip.in_frame, clip.get_out_frame())
        self._edit_total = len(self._edit_cache)
        self._edit_frame_no = 0

        preset = GRADIENT_PRESETS[0]
        self._edit_swings = [TrajectoryData(
            color_start_hex=preset[0], color_end_hex=preset[1], thickness=3
        )]
        self._edit_swing_idx = 0

        self.edit_slider.configure(to=max(self._edit_total - 1, 1))
        self._edit_update_color_buttons()
        self._edit_update_display()
        self._refresh_edit_clips_list()
        self.tabview.set("編集")

    def _on_playout_frame(self, frame, frame_no, total):
        """送出プレビュー更新 — 再生スレッドから呼ばれる

        最新フレームのみ保持。重い処理は再生タイミングに影響するため
        ここでは保存のみ行い、GUI側で処理する。
        """
        self._po_pending_frame = (frame, frame_no, total)
        if not self._po_gui_scheduled:
            self._po_gui_scheduled = True
            self.after(0, self._po_flush_frame)

    def _po_flush_frame(self):
        """最新フレームをGUIに反映"""
        self._po_gui_scheduled = False
        pending = self._po_pending_frame
        if pending is None:
            return
        self._po_pending_frame = None
        frame, frame_no, total = pending
        cw = self.playout_canvas.winfo_width()
        ch = self.playout_canvas.winfo_height()
        if cw > 10 and ch > 10:
            self._playout_photo, _ = frame_to_photo(frame, cw, ch)
            self.playout_canvas.delete("all")
            self.playout_canvas.create_image(
                cw // 2, ch // 2, anchor="center", image=self._playout_photo)
        self.po_status.configure(
            text=f"▶ {frame_no}/{total}", text_color="#00FF00")
        # シークバー更新
        self._po_slider_updating = True
        self.po_seek_slider.configure(to=max(total - 1, 1))
        self.po_seek_slider.set(frame_no)
        self.po_frame_label.configure(text=f"{frame_no} / {total - 1}")
        self._po_slider_updating = False

    def _on_playout_clip_changed(self, index, clip):
        """再生中にクリップが切り替わった時"""
        def update():
            self._playout_selected_idx = index
            self._playout_highlight_row(index)
            total = clip.get_duration_frames()
            self._po_slider_updating = True
            self.po_seek_slider.configure(to=max(total - 1, 1))
            self.po_seek_slider.set(0)
            self.po_frame_label.configure(text=f"0 / {total - 1}")
            self._po_slider_updating = False
        self.after(0, update)

    def _on_playout_ended(self):
        def update():
            # current_indexはそのまま維持 (再生終了位置に留まる)
            self.po_status.configure(text="⏹ STOP", text_color="white")
            self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                        hover_color="#228B22")
            self._shuttle_playing_by_shuttle = False
        self.after(0, update)

    # =========================================================================
    # 設定タブ
    # =========================================================================
    def _build_settings_tab(self):
        tab = self.tab_settings
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="システム設定", font=("", 20, "bold")).pack(pady=15)

        # プロジェクトフォルダ
        sec1 = ctk.CTkFrame(tab)
        sec1.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(sec1, text="プロジェクトフォルダ").pack(anchor="w", padx=10, pady=(5, 0))
        dir_row = ctk.CTkFrame(sec1, fg_color="transparent")
        dir_row.pack(fill="x", padx=10, pady=5)
        self.project_dir_entry = ctk.CTkEntry(dir_row, width=400)
        self.project_dir_entry.insert(0, str(self.project_dir))
        self.project_dir_entry.pack(side="left", padx=(0, 5))
        ctk.CTkButton(dir_row, text="変更", width=80,
                       command=self._change_project_dir).pack(side="left")

        # 録画フォルダ
        sec2 = ctk.CTkFrame(tab)
        sec2.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(sec2, text="録画保存フォルダ").pack(anchor="w", padx=10, pady=(5, 0))
        rec_row = ctk.CTkFrame(sec2, fg_color="transparent")
        rec_row.pack(fill="x", padx=10, pady=5)
        self.record_dir_entry = ctk.CTkEntry(rec_row, width=400)
        self.record_dir_entry.insert(0, self.settings["record_dir"])
        self.record_dir_entry.pack(side="left", padx=(0, 5))
        ctk.CTkButton(rec_row, text="変更", width=80,
                       command=self._change_record_dir).pack(side="left")

        # 解像度・FPS
        sec3 = ctk.CTkFrame(tab)
        sec3.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(sec3, text="映像設定").pack(anchor="w", padx=10, pady=(5, 0))
        vid_row = ctk.CTkFrame(sec3, fg_color="transparent")
        vid_row.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(vid_row, text="解像度:").pack(side="left", padx=(0, 5))
        self.resolution_combo = ctk.CTkComboBox(
            vid_row, values=["1920x1080", "1280x720", "3840x2160"], width=150
        )
        self.resolution_combo.set(f"{self.settings['width']}x{self.settings['height']}")
        self.resolution_combo.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(vid_row, text="FPS:").pack(side="left", padx=(0, 5))
        self.fps_combo = ctk.CTkComboBox(
            vid_row, values=["29.97", "25", "30", "50", "59.94", "60"], width=100
        )
        self.fps_combo.set(str(self.settings["fps"]))
        self.fps_combo.pack(side="left")

        # デバイス
        sec4 = ctk.CTkFrame(tab)
        sec4.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(sec4, text="DeckLink デバイス").pack(anchor="w", padx=10, pady=(5, 0))

        devices = enumerate_decklink_devices()
        device_names = [d.name for d in devices] if devices else ["(デバイスなし - フォールバックモード)"]

        dev_row = ctk.CTkFrame(sec4, fg_color="transparent")
        dev_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(dev_row, text="入力:").pack(side="left", padx=(0, 5))
        self.input_dev_combo = ctk.CTkComboBox(dev_row, values=device_names, width=250)
        self.input_dev_combo.pack(side="left", padx=(0, 20))
        ctk.CTkLabel(dev_row, text="出力:").pack(side="left", padx=(0, 5))
        self.output_dev_combo = ctk.CTkComboBox(dev_row, values=device_names, width=250)
        self.output_dev_combo.pack(side="left")

        # ShuttlePRO v2 ボタン設定
        self._build_shuttle_settings(tab)

        # 保存ボタン
        ctk.CTkButton(tab, text="設定を保存", width=200, height=40,
                       font=("", 14, "bold"),
                       fg_color="#006400", hover_color="#228B22",
                       command=self._save_settings).pack(pady=20)

    def _build_shuttle_settings(self, tab):
        """ShuttlePRO v2 ボタン設定セクション"""
        sec = ctk.CTkFrame(tab)
        sec.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(sec, text="ShuttlePRO v2 ボタン設定",
                      font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        ctk.CTkLabel(sec, text="ボタンを押すと●が点灯します（番号確認用）",
                      font=("", 10), text_color="#888").pack(anchor="w", padx=10, pady=(0, 5))

        content = ctk.CTkFrame(sec, fg_color="transparent")
        content.pack(fill="x", padx=5, pady=5)

        # 左: デバイス図 (インタラクティブ)
        diagram_frame = ctk.CTkFrame(content)
        diagram_frame.pack(side="left", padx=5, pady=5)
        self._shuttle_canvas = Canvas(diagram_frame, width=260, height=400,
                                       bg="#1a1a1a", highlightthickness=0)
        self._shuttle_canvas.pack(padx=8, pady=8)
        self._shuttle_btn_ovals = {}    # pos_idx -> oval id
        self._shuttle_btn_texts = {}    # pos_idx -> text id
        self._shuttle_pos_colors = {}   # pos_idx -> default fill
        self._shuttle_learn_pos = None  # クリックで選択中のポジション
        # ポジション→HIDボタン番号マッピング (設定から復元)
        saved = self.settings.data.get("shuttle_pos_mapping", None)
        if saved:
            self._shuttle_pos_map = {int(k): int(v) for k, v in saved.items()}
        else:
            self._shuttle_pos_map = {i: i for i in range(1, 16)}
        self._draw_shuttle_diagram()
        self._shuttle_learn_label = ctk.CTkLabel(
            diagram_frame, text="位置をクリック → ボタンを押す",
            font=("", 10), text_color="#888")
        self._shuttle_learn_label.pack(pady=(0, 5))

        # 右: ボタン割り当て (2列)
        assign_outer = ctk.CTkFrame(content)
        assign_outer.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        btn_map = self.settings.data.get("shuttle_buttons", {})
        self._shuttle_btn_combos = {}
        self._shuttle_btn_indicators = {}

        col_frame = ctk.CTkFrame(assign_outer, fg_color="transparent")
        col_frame.pack(fill="both", expand=True, padx=5, pady=5)
        col_frame.grid_columnconfigure(0, weight=1)
        col_frame.grid_columnconfigure(1, weight=1)

        # 2列に分けて配置 (1-8 左列, 9-15 右列)
        for i in range(1, 16):
            col = 0 if i <= 8 else 1
            row_idx = (i - 1) if i <= 8 else (i - 9)

            row = ctk.CTkFrame(col_frame, fg_color="transparent")
            row.grid(row=row_idx, column=col, sticky="ew", padx=3, pady=2)

            indicator = ctk.CTkLabel(row, text="●", width=18,
                                      text_color="#333", font=("", 12))
            indicator.pack(side="left")
            self._shuttle_btn_indicators[i] = indicator

            ctk.CTkLabel(row, text=f"{i:2d}:", width=30,
                          font=("", 11, "bold")).pack(side="left")

            combo = ctk.CTkComboBox(row, values=SHUTTLE_ACTION_LABELS,
                                     width=130, font=("", 11))
            current_action = btn_map.get(str(i), "none")
            label = "なし"
            for key, lbl in SHUTTLE_ACTIONS:
                if key == current_action:
                    label = lbl
                    break
            combo.set(label)
            combo.pack(side="left", padx=3)
            self._shuttle_btn_combos[i] = combo

    def _draw_shuttle_diagram(self):
        """ShuttlePRO v2 デバイス図を描画 (インタラクティブ)

        位置をクリック → 物理ボタンを押す → 番号が配置される。
        """
        c = self._shuttle_canvas
        c.delete("all")
        # デバイス本体 (卵型)
        c.create_oval(25, 8, 235, 392, fill="#2a2a2a", outline="#555", width=2)

        silver = "#b8b8b8"
        dark = "#444444"
        ol = "#777"

        # 物理ポジション定義: (pos_idx, x, y, rx, ry, default_fill)
        positions = [
            # 上段4個 (小丸)
            (1, 75, 48, 11, 11, silver),
            (2, 112, 42, 11, 11, silver),
            (3, 150, 42, 11, 11, silver),
            (4, 188, 48, 11, 11, silver),
            # 中段5個 (やや大)
            (5, 48, 97, 14, 14, silver),
            (6, 90, 92, 14, 14, silver),
            (7, 130, 90, 14, 14, silver),
            (8, 170, 92, 14, 14, silver),
            (9, 212, 97, 14, 14, silver),
            # ジョグ下 左3個
            (10, 55, 298, 16, 10, dark),
            (11, 55, 325, 16, 10, silver),
            (12, 55, 352, 16, 10, silver),
            # ジョグ下 右3個
            (13, 205, 298, 16, 10, dark),
            (14, 205, 325, 16, 10, silver),
            (15, 205, 352, 16, 10, silver),
        ]

        self._shuttle_btn_ovals.clear()
        self._shuttle_btn_texts.clear()
        self._shuttle_pos_colors.clear()

        for pos_idx, x, y, rx, ry, fill in positions:
            btn_no = self._shuttle_pos_map.get(pos_idx, pos_idx)
            text_col = "#CCC" if fill == dark else "#222"
            tag = f"spos_{pos_idx}"
            oval = c.create_oval(x - rx, y - ry, x + rx, y + ry,
                                  fill=fill, outline=ol, width=1, tags=(tag,))
            txt = c.create_text(x, y, text=str(btn_no),
                                 font=("", 9, "bold"), fill=text_col, tags=(tag,))
            c.tag_bind(tag, "<Button-1>",
                       lambda e, p=pos_idx: self._on_shuttle_pos_click(p))
            self._shuttle_btn_ovals[pos_idx] = oval
            self._shuttle_btn_texts[pos_idx] = txt
            self._shuttle_pos_colors[pos_idx] = fill

        # シャトルリング
        cx, cy = 130, 200
        c.create_oval(cx - 68, cy - 68, cx + 68, cy + 68,
                       fill="#1a1a1a", outline="#666", width=3)
        c.create_text(cx, cy - 78, text="SHUTTLE RING",
                       font=("", 8), fill="#666")
        # ジョグダイヤル
        c.create_oval(cx - 38, cy - 38, cx + 38, cy + 38,
                       fill="#3a3a3a", outline="#888", width=2)
        c.create_text(cx, cy, text="JOG",
                       font=("", 11, "bold"), fill="#999")

    def _on_shuttle_pos_click(self, pos_idx):
        """Canvas上のポジションをクリック → 物理ボタン待ちモード"""
        c = self._shuttle_canvas
        # 前の選択をリセット
        if self._shuttle_learn_pos is not None:
            prev = self._shuttle_learn_pos
            c.itemconfig(self._shuttle_btn_ovals[prev],
                         fill=self._shuttle_pos_colors[prev])
        # 選択ポジションをハイライト (黄色)
        self._shuttle_learn_pos = pos_idx
        c.itemconfig(self._shuttle_btn_ovals[pos_idx], fill="#FFFF00")
        self._shuttle_learn_label.configure(
            text=f"位置 {pos_idx}: このボタンを押してください",
            text_color="#FFFF00")

    def _flash_shuttle_btn(self, btn_no):
        """ボタン押下時の処理

        Learn mode: クリック済みのCanvasポジションにボタン番号を割り当て
        通常: Canvas上とリスト側インジケータを緑に点灯
        """
        c = self._shuttle_canvas

        # --- Learn mode: ポジション待ち中なら割り当て ---
        if self._shuttle_learn_pos is not None:
            pos = self._shuttle_learn_pos
            self._shuttle_pos_map[pos] = btn_no
            # Canvas上のテキストを更新
            c.itemconfig(self._shuttle_btn_texts[pos], text=str(btn_no))
            # 緑に光らせてから元の色に戻す
            c.itemconfig(self._shuttle_btn_ovals[pos], fill="#00FF00")
            restore = self._shuttle_pos_colors[pos]
            self.after(600, lambda: c.itemconfig(
                self._shuttle_btn_ovals[pos], fill=restore))
            self._shuttle_learn_pos = None
            self._shuttle_learn_label.configure(
                text=f"BTN {btn_no} → 位置 {pos} に設定",
                text_color="#00FF00")
            self.after(2000, lambda: self._shuttle_learn_label.configure(
                text="位置をクリック → ボタンを押す", text_color="#888"))
            return

        # --- 通常モード: 対応するポジションを点灯 ---
        # pos_map の逆引き: btn_no → pos_idx
        for pos_idx, mapped_btn in self._shuttle_pos_map.items():
            if mapped_btn == btn_no and pos_idx in self._shuttle_btn_ovals:
                oval = self._shuttle_btn_ovals[pos_idx]
                c.itemconfig(oval, fill="#00FF00")
                restore = self._shuttle_pos_colors[pos_idx]
                self.after(800, lambda o=oval, r=restore: c.itemconfig(o, fill=r))
                break

        # リスト側インジケータ点灯
        if btn_no in self._shuttle_btn_indicators:
            ind = self._shuttle_btn_indicators[btn_no]
            ind.configure(text_color="#00FF00")
            self.after(800, lambda: ind.configure(text_color="#333"))

    def _change_project_dir(self):
        d = filedialog.askdirectory(title="プロジェクトフォルダを選択")
        if d:
            self.project_dir_entry.delete(0, "end")
            self.project_dir_entry.insert(0, d)

    def _change_record_dir(self):
        d = filedialog.askdirectory(title="録画保存フォルダを選択")
        if d:
            self.record_dir_entry.delete(0, "end")
            self.record_dir_entry.insert(0, d)

    def _save_settings(self):
        self.settings["project_dir"] = self.project_dir_entry.get()
        self.settings["record_dir"] = self.record_dir_entry.get()

        res = self.resolution_combo.get()
        if "x" in res:
            w, h = res.split("x")
            self.settings["width"] = int(w)
            self.settings["height"] = int(h)

        try:
            self.settings["fps"] = float(self.fps_combo.get())
        except ValueError:
            pass

        # キャプチャモード
        self.settings["capture_mode"] = self._get_current_capture_mode().value

        # ShuttlePRO ボタンマッピング
        btn_map = {}
        for i in range(1, 16):
            combo = self._shuttle_btn_combos.get(i)
            if combo:
                label = combo.get()
                action_key = "none"
                for key, lbl in SHUTTLE_ACTIONS:
                    if lbl == label:
                        action_key = key
                        break
                btn_map[str(i)] = action_key
        self.settings["shuttle_buttons"] = btn_map

        # ShuttlePRO ポジションマッピング
        self.settings["shuttle_pos_mapping"] = {
            str(k): v for k, v in self._shuttle_pos_map.items()
        }

        self.settings.save()
        self.recorder = Recorder(
            self.settings["record_dir"],
            self.settings["width"],
            self.settings["height"],
            self.settings["fps"],
        )
        print("[Settings] 設定を保存しました")

    # =========================================================================
    # ShuttlePRO v2
    # =========================================================================
    def _setup_shuttle(self):
        """ShuttlePRO v2 のイベントハンドラを設定"""
        self._shuttle_playing_by_shuttle = False
        self._shuttle_reverse_id = None  # after() ID for reverse playback
        self._shuttle_reverse_step = 0   # current reverse step size

        def on_jog(delta):
            self.after(0, lambda d=delta: self._shuttle_jog(d))

        def on_shuttle(position):
            self.after(0, lambda p=position: self._shuttle_ring(p))

        def on_button(btn, pressed):
            self.after(0, lambda b=btn, p=pressed: self._shuttle_button_event(b, p))

        self.shuttle.on_jog = on_jog
        self.shuttle.on_shuttle = on_shuttle
        self.shuttle.on_button = on_button

        # 接続状態チェック (1秒おき)
        def check_connection():
            if self.shuttle.connected:
                self.po_shuttle_status.configure(
                    text="接続済", text_color="#00FF00")
            else:
                self.po_shuttle_status.configure(
                    text="未接続", text_color="#888")
            self.after(2000, check_connection)
        self.after(2000, check_connection)

    def _shuttle_jog(self, delta):
        """ジョグ: フレームステップ (再生中は一時停止してからステップ)"""
        self.po_shuttle_info.configure(text=f"JOG {'+' if delta > 0 else ''}{delta}")
        current_tab = self.tabview.get()
        if current_tab == "送出":
            # 再生中なら一時停止してからフレーム送り
            if self.playout._playing and not self.playout._paused:
                self.playout.pause()
                self.po_status.configure(text="⏸ PAUSED", text_color="#FFAA00")
                self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                            hover_color="#228B22")
            self._playout_seek_delta(delta)
        elif current_tab == "編集":
            self._edit_jump(delta)

    def _shuttle_ring(self, position):
        """シャトルリング: 可変速再生

        position: -7(全CCW) ~ 0(センター) ~ +7(全CW)
          0     : 一時停止
          +1~+3 : スロー再生 (1/8, 1/4, 1/2)
          +4    : 通常速 (1x)
          +5~+7 : 高速 (将来用、現在は1x)
          -1~-7 : 逆方向は現在未対応、フレームバック
        """
        current_tab = self.tabview.get()
        if current_tab != "送出":
            return

        if position == 0:
            # センター: 一時停止
            self._stop_reverse_timer()
            if self.playout._playing and not self.playout._paused:
                self.playout.pause()
                self.po_status.configure(text="⏸ PAUSED", text_color="#FFAA00")
                self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                            hover_color="#228B22")
            self._shuttle_playing_by_shuttle = False
        elif position > 0:
            # CW: 前方再生 (速度マップ)
            self._stop_reverse_timer()
            speed_map = {1: 0.125, 2: 0.25, 3: 0.5, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}
            speed_labels = {1: "1/8", 2: "1/4", 3: "1/2", 4: "1x", 5: "1x", 6: "1x", 7: "1x"}
            spd = speed_map.get(position, 1.0)
            self._po_speed = spd
            self.playout.speed = spd
            self.po_speed_seg.set(speed_labels.get(position, "1x"))

            if not self.playout._playing:
                # 再生可能なクリップがあるか確認
                if self.playout.current_index >= len(self.playout.playlist):
                    return
                self._playout_play()
                self.po_play_btn.configure(text="⏸ PAUSE", fg_color="#B8860B",
                                            hover_color="#DAA520")
            elif self.playout._paused:
                self.playout.play()
                self.po_play_btn.configure(text="⏸ PAUSE", fg_color="#B8860B",
                                            hover_color="#DAA520")
            self._update_play_status()
            self._shuttle_playing_by_shuttle = True
        else:
            # CCW: 逆方向連続再生 (タイマーベース)
            # play_loopが動いているとDeckLink出力・self._capアクセスが競合するので
            # 完全停止してからリバース送りを行う (paused中のself._capアクセスは
            # 再生スレッドとレースし、プレビューが更新されない問題があった)
            if self.playout._playing:
                # 現在位置を保持 (stopは_current_frame_noをリセットしない)
                saved_frame = self.playout._current_frame_no
                self.playout.stop()
                # 再生スレッド終了待ち (短時間)
                if self.playout._thread and self.playout._thread.is_alive():
                    self.playout._thread.join(timeout=0.3)
                self.playout._thread = None
                self.playout._current_frame_no = saved_frame
                # capが閉じていれば再オープン
                if self.playout._cap is None and 0 <= self.playout.current_index < len(self.playout.playlist):
                    self.playout._open_cap(self.playout.current_index)
                    self.playout._current_frame_no = saved_frame
                self.po_status.configure(text="◀ REV", text_color="#00CCFF")
                self.po_play_btn.configure(text="▶ PLAY", fg_color="#006400",
                                            hover_color="#228B22")
            # ステップ量とインターバル (位置が大きいほど速い)
            rev_map = {
                -1: (1, 120),   # 1F every 120ms (~1/8x)
                -2: (1, 60),    # 1F every 60ms  (~1/4x)
                -3: (1, 30),    # 1F every 30ms  (~1/2x)
                -4: (1, 15),    # 1F every 15ms  (~1x)
                -5: (2, 15),    # 2F every 15ms  (~2x)
                -6: (3, 15),    # 3F every 15ms  (~3x)
                -7: (5, 15),    # 5F every 15ms  (~5x)
            }
            step, interval = rev_map.get(position, (1, 120))
            self._shuttle_reverse_step = step
            self._start_reverse_timer(step, interval)
            self.po_status.configure(text="◀ REV", text_color="#00CCFF")

    def _start_reverse_timer(self, step, interval_ms):
        """逆再生タイマー開始 (既存タイマーは停止してから)"""
        self._stop_reverse_timer()
        self._shuttle_reverse_step = step

        def tick():
            if self._shuttle_reverse_step == 0:
                return
            self._playout_seek_delta(-self._shuttle_reverse_step)
            self._shuttle_reverse_id = self.after(interval_ms, tick)

        # 初回即実行
        self._playout_seek_delta(-step)
        self._shuttle_reverse_id = self.after(interval_ms, tick)

    def _stop_reverse_timer(self):
        """逆再生タイマー停止"""
        if self._shuttle_reverse_id is not None:
            self.after_cancel(self._shuttle_reverse_id)
            self._shuttle_reverse_id = None
        self._shuttle_reverse_step = 0

    def _shuttle_button_event(self, btn, pressed):
        """ShuttlePRO v2 ボタンイベント (press/release両方)"""
        action_label = "PRESS" if pressed else "release"
        self.po_shuttle_info.configure(text=f"BTN {btn} {action_label}")

        # 押下時にインジケータ点灯 (どのタブでも)
        if pressed:
            self._flash_shuttle_btn(btn)

        if not pressed:
            return

        current_tab = self.tabview.get()
        if current_tab != "送出":
            return

        # 設定からアクションを取得
        btn_map = self.settings.data.get("shuttle_buttons", {})
        action = btn_map.get(str(btn), "none")

        action_dispatch = {
            "play_pause": self._playout_toggle_play,
            "play": lambda: self._playout_play(),
            "stop": self._playout_stop,
            "cue": self._playout_cue_top,
            "prev": self._playout_prev,
            "next": self._playout_next,
            "speed_1x": lambda: (self.po_speed_seg.set("1x"), self._on_speed_change("1x")),
            "speed_1_2": lambda: (self.po_speed_seg.set("1/2"), self._on_speed_change("1/2")),
            "speed_1_4": lambda: (self.po_speed_seg.set("1/4"), self._on_speed_change("1/4")),
            "speed_1_8": lambda: (self.po_speed_seg.set("1/8"), self._on_speed_change("1/8")),
            "frame_fwd_1": lambda: self._playout_seek_delta(1),
            "frame_back_1": lambda: self._playout_seek_delta(-1),
            "frame_fwd_5": lambda: self._playout_seek_delta(5),
            "frame_back_5": lambda: self._playout_seek_delta(-5),
        }
        fn = action_dispatch.get(action)
        if fn:
            fn()

    # =========================================================================
    # グローバルキーバインド
    # =========================================================================
    def _bind_global_keys(self):
        import tkinter as tk

        def on_key(event):
            w = event.widget
            if isinstance(w, (tk.Entry, ctk.CTkEntry)):
                return

            key = event.keysym

            current_tab = self.tabview.get()

            # クリップタブ
            if current_tab == "クリップ":
                if key.lower() == "d" or key == "Right":
                    self._clip_jump(self._frame_step)
                elif key.lower() == "a" or key == "Left":
                    self._clip_jump(-self._frame_step)
                elif key.lower() == "w":
                    self._clip_jump(self._frame_step * 5)
                elif key.lower() == "s":
                    self._clip_jump(-self._frame_step * 5)
                elif key.lower() == "i":
                    self._set_in_current()
                elif key.lower() == "o":
                    self._set_out_current()
                return

            # 編集タブ
            if current_tab == "編集":
                if key.lower() == "d" or key == "Right":
                    self._edit_jump(self._frame_step)
                elif key.lower() == "a" or key == "Left":
                    self._edit_jump(-self._frame_step)
                elif key.lower() == "w":
                    self._edit_jump(self._frame_step * 5)
                elif key.lower() == "s":
                    self._edit_jump(-self._frame_step * 5)
                elif key == "space":
                    self._edit_toggle_play()
                return

            # 送出タブ (プロ用ショートカット)
            if current_tab == "送出":
                if key == "space":
                    self._playout_toggle_play()
                elif key == "Return" or key == "F5":
                    self._playout_play()
                    self.po_play_btn.configure(text="⏸ PAUSE",
                        fg_color="#B8860B", hover_color="#DAA520")
                elif key == "Escape":
                    self._playout_cue_top()
                elif key == "F8" or key.lower() == "n":
                    self._playout_next()
                elif key == "F1" or key.lower() == "p":
                    self._playout_prev()
                elif key.lower() == "d" or key == "Right":
                    self._playout_seek_delta(self._frame_step)
                elif key.lower() == "a" or key == "Left":
                    self._playout_seek_delta(-self._frame_step)
                elif key.lower() == "w":
                    self._playout_seek_delta(self._frame_step * 5)
                elif key.lower() == "s":
                    self._playout_seek_delta(-self._frame_step * 5)
                elif key == "1":
                    self.po_speed_seg.set("1x")
                    self._on_speed_change("1x")
                elif key == "2":
                    self.po_speed_seg.set("1/2")
                    self._on_speed_change("1/2")
                elif key == "3":
                    self.po_speed_seg.set("1/4")
                    self._on_speed_change("1/4")
                elif key == "4":
                    self.po_speed_seg.set("1/8")
                    self._on_speed_change("1/8")
                return

            # 収録タブ
            if current_tab == "収録":
                if key == "F9" or key == "space":
                    self._toggle_rec()
                return

        self.bind_all("<Key>", on_key)

    # =========================================================================
    # 終了処理
    # =========================================================================
    def _on_close(self):
        print("[App] 終了処理...")
        self.shuttle.stop()
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        # 録画書き込みスレッドを停止 (キューを空にしてから終了)
        self._capture_write_running = False
        if self._capture_write_thread and self._capture_write_thread.is_alive():
            self._capture_write_thread.join(timeout=3.0)
        if self.deck_input:
            self.deck_input.stop()
        if self.deck_output:
            self.deck_output.stop()
        # 再生スレッドを確実に停止
        self.playout.stop()
        self.playout._wait_thread()
        self.playout.save_playlist(self._playout_json)
        self.clip_manager.save()
        self.destroy()


# =============================================================================
# エントリポイント
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Golf Swing Broadcast System")
    parser.add_argument("--project", type=str, default=None,
                        help="プロジェクトフォルダパス")
    args = parser.parse_args()

    app = GolfBroadcastApp(project_dir=args.project)
    app.mainloop()


if __name__ == "__main__":
    main()
