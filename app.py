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
import collections
import datetime
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter
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
from ffmpeg_writer import FFmpegWriter, find_ffmpeg

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

# キーボードショートカットアクション定義
# (action_id, label, default_key, tab_context)
KEYBOARD_ACTIONS = [
    # 共通 (クリップ・編集・送出)
    ("frame_fwd",       "+1F",              "d",       "共通"),
    ("frame_back",      "-1F",              "a",       "共通"),
    ("frame_fwd_fast",  "+5F",              "w",       "共通"),
    ("frame_back_fast", "-5F",              "s",       "共通"),
    ("step_1",          "ステップ 1F",      "F2",      "共通"),
    ("step_2",          "ステップ 2F",      "F3",      "共通"),
    ("step_5",          "ステップ 5F",      "F4",      "共通"),
    ("step_10",         "ステップ 10F",     "F6",      "共通"),
    # クリップ
    ("set_in",          "IN点設定",         "i",       "クリップ"),
    ("set_out",         "OUT点設定",        "o",       "クリップ"),
    # 編集
    ("edit_play",       "再生/停止 (編集)", "space",   "編集"),
    ("edit_set_in",     "IN点設定 (編集)",  "i",       "編集"),
    ("edit_set_out",    "OUT点設定 (編集)", "o",       "編集"),
    ("zoom_reset",      "ズームリセット",   "Home",    "編集"),
    # 送出
    ("po_play_pause",   "PLAY/PAUSE",       "space",   "送出"),
    ("po_play",         "PLAY",             "Return",  "送出"),
    ("po_cue_top",      "CUE (頭出し)",     "Escape",  "送出"),
    ("po_next",         "次クリップ",       "n",       "送出"),
    ("po_prev",         "前クリップ",       "p",       "送出"),
    ("po_speed_1x",     "速度 1x",          "1",       "送出"),
    ("po_speed_1_2",    "速度 1/2",         "2",       "送出"),
    ("po_speed_1_4",    "速度 1/4",         "3",       "送出"),
    ("po_speed_1_8",    "速度 1/8",         "4",       "送出"),
    # 収録
    ("toggle_rec",      "REC/STOP",         "F9",      "収録"),
]
DEFAULT_KEYBOARD_SHORTCUTS = {a[0]: a[2] for a in KEYBOARD_ACTIONS}

# keysym 表示名マッピング
KEYSYM_DISPLAY = {
    "space": "Space", "Return": "Enter", "Escape": "Esc",
    "Left": "←", "Right": "→", "Up": "↑", "Down": "↓",
    "Home": "Home", "End": "End",
    "BackSpace": "BS", "Delete": "Del", "Tab": "Tab",
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
    fade_frames>0 の場合、最初の点出現から fade_frames かけてフェードイン。
    end_frame が設定されていれば、その付近で fade_frames かけてフェードアウト。
    end_frame 未設定 (-1) の場合はフェードアウトしない (軌道を残す)。
    """
    fade = getattr(swing, "fade_frames", 0)
    end_f = getattr(swing, "end_frame", -1)
    if not swing.points:
        return 0.0

    first_f = swing.points[0][2]

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

    # フェードアウト (end_frame 設定時のみ)
    if end_f >= 0:
        if current_frame <= end_f:
            fo_ratio = 1.0
        else:
            fo_ratio = max(0.0, 1.0 - (current_frame - end_f) / fade)
    else:
        # end_frame 未設定: フェードアウトしない
        fo_ratio = 1.0

    return base_alpha * min(fi_ratio, fo_ratio)


# =============================================================================
# フレームキャッシュ
# =============================================================================
class FrameCache:
    """JPEG圧縮フレームキャッシュ (メモリ効率版)

    フレームをJPEG圧縮して保持し、取得時にデコードする。
    BGR生フレーム (~6MB/枚) → JPEG (~200KB/枚) で約30倍のメモリ節約。
    直近アクセスされたフレームはデコード済みLRUキャッシュに保持し、
    次フレームを先読みデコードして連続コマ送りを高速化する。
    """
    _JPEG_QUALITY = 92
    _LRU_SIZE = 30   # デコード済みフレームのLRU保持数 (~180MB)
    _PREFETCH = 3    # 先読みフレーム数

    def __init__(self, video_path, in_frame=0, out_frame=-1):
        self._frames_jpg = []  # JPEG bytes のリスト
        self._total_expected = 0
        self._loading = False
        self._video_path = str(video_path)
        self._in_frame = in_frame
        # デコード済みLRUキャッシュ
        self._lru_cache = collections.OrderedDict()
        self._lru_lock = threading.Lock()
        self._prefetching = set()  # 先読み中のインデックス

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
            self._frames_jpg.append(self._encode(f))
        cap.release()

    @staticmethod
    def _encode(frame):
        _, buf = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, FrameCache._JPEG_QUALITY])
        return buf.tobytes()

    @staticmethod
    def _decode(jpg_bytes):
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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
                self._frames_jpg.append(self._encode(f))
            cap.release()
            self._loading = False
            if on_done:
                on_done()

        threading.Thread(target=_load, daemon=True).start()

    def __len__(self):
        return self._total_expected if self._total_expected > 0 else len(self._frames_jpg)

    def _lru_put(self, idx, frame):
        """LRUキャッシュにフレームを格納 (スレッドセーフ)"""
        with self._lru_lock:
            self._lru_cache[idx] = frame
            if len(self._lru_cache) > self._LRU_SIZE:
                self._lru_cache.popitem(last=False)

    def _prefetch(self, current_idx):
        """隣接フレームをバックグラウンドで先読みデコード"""
        targets = []
        for d in range(1, self._PREFETCH + 1):
            for idx in (current_idx + d, current_idx - d):
                if 0 <= idx < len(self._frames_jpg) and idx not in self._lru_cache:
                    targets.append(idx)
        if not targets:
            return

        def _do_prefetch():
            for idx in targets:
                if idx in self._prefetching:
                    continue
                self._prefetching.add(idx)
                try:
                    with self._lru_lock:
                        if idx in self._lru_cache:
                            continue
                    if 0 <= idx < len(self._frames_jpg):
                        frame = self._decode(self._frames_jpg[idx])
                        self._lru_put(idx, frame)
                finally:
                    self._prefetching.discard(idx)

        threading.Thread(target=_do_prefetch, daemon=True).start()

    def __getitem__(self, idx):
        # LRUキャッシュにヒットすれば即返却 (読み取り専用 — 変更禁止)
        with self._lru_lock:
            if idx in self._lru_cache:
                self._lru_cache.move_to_end(idx)
                self._prefetch(idx)
                return self._lru_cache[idx]
        if 0 <= idx < len(self._frames_jpg):
            frame = self._decode(self._frames_jpg[idx])
            self._lru_put(idx, frame)
            self._prefetch(idx)
            return frame
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
            "crf": 18,             # 録画・書き出し品質 (0=ロスレス, 18=高品質, 23=標準, 28=低品質)
            "growing_buffer_sec": 60,  # グローウィングバッファ最大秒数
            "trajectory_style": {  # 最後に使った軌道スタイル
                "color_start_hex": "#FFFF00",
                "color_end_hex": "#FF0000",
                "thickness": 3,
                "blur": 0,
                "fade_frames": 0,
                "alpha": 0.85,
            },
            "keyboard_shortcuts": dict(DEFAULT_KEYBOARD_SHORTCUTS),
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
    BGR (3ch) / BGRA (4ch) 両対応。
    """
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        code = cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB
        rgb = cv2.cvtColor(frame, code)
        return ImageTk.PhotoImage(Image.fromarray(rgb)), 1.0
    scale = min(max_w / w, max_h / h)
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    code = cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB
    rgb = cv2.cvtColor(frame, code)
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
        # 1) デフォルトor引数のディレクトリで設定を読み込む
        init_dir = Path(project_dir or DEFAULT_PROJECT_DIR)
        init_dir.mkdir(parents=True, exist_ok=True)
        self.settings = AppSettings(init_dir)

        # 2) 保存済みの project_dir があればそちらを使う (--project 指定時は引数優先)
        if project_dir is None and self.settings["project_dir"] != str(init_dir):
            saved_dir = Path(self.settings["project_dir"])
            if saved_dir.exists() or saved_dir.parent.exists():
                self.project_dir = saved_dir
                self.project_dir.mkdir(parents=True, exist_ok=True)
                # 新フォルダの settings.json があればそこから全設定を再読み込み
                new_settings_path = self.project_dir / "settings.json"
                if new_settings_path.exists():
                    self.settings = AppSettings(self.project_dir)
                else:
                    self.settings.path = new_settings_path
            else:
                self.project_dir = init_dir
        else:
            self.project_dir = init_dir

        self.clip_manager = ClipManager(str(self.project_dir))

        # HWエンコーダを事前検出 (初回書き出しの待ち時間を削減)
        threading.Thread(target=self._predetect_hw_encoder, daemon=True).start()

        # デバイス
        self.deck_input = None
        self.deck_output = None
        self.recorder = Recorder(
            self.settings["record_dir"],
            self.settings["width"],
            self.settings["height"],
            self.settings["fps"],
            crf=self.settings["crf"],
            growing_buffer_sec=self.settings["growing_buffer_sec"],
        )

        # 送出エンジン
        self.playout = PlayoutEngine()
        self._playout_json = str(self.project_dir / "playout.json")
        self._exports_dir = self.project_dir / "exports"
        self._exports_dir.mkdir(parents=True, exist_ok=True)
        self.playout.load_playlist(self._playout_json)
        self.playout.scan_directory(str(self._exports_dir))
        self.playout.save_playlist(self._playout_json)

        # 録画書き込みキュー
        # DeckLink COMコールバックスレッドを録画I/Oから分離し、カクツキを防止する。
        # コールバックはフレームをキューに入れるだけ (< 1ms)、
        # 別スレッドが MP4書き込み + JPEGエンコードを担当する。
        self._capture_queue: queue.Queue = queue.Queue(maxsize=600)
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

        # グローウィング再書き出しキュー: [(rec_path, in_f, out_f, clip_path, fps), ...]
        self._growing_reexport_queue = []

        # ダーティフラグ: タブ切替時の無駄なスキャンを抑制
        self._clips_tab_dirty = True   # クリップタブ再構築が必要
        self._edit_tab_dirty = True    # 編集タブクリップリスト再構築が必要
        self._playout_dirty = True     # 送出リスト再スキャンが必要

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
        """タブ切り替え時の処理 (ファイルシステムスキャンは行わない)"""
        current = self.tabview.get()
        if current == "クリップ":
            if self._clips_tab_dirty:
                self._refresh_clips_list(scan=False)
        elif current == "編集":
            if self._edit_tab_dirty:
                self._refresh_edit_clips_list()
                self._edit_tab_dirty = False
        elif current == "送出":
            # 編集中の軌道を自動保存
            self._edit_autosave_trajectory()
            if self._playout_dirty:
                self._refresh_playout_list()
                self._playout_dirty = False

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
                self._capture_enqueued = getattr(self, '_capture_enqueued', 0) + 1
            except queue.Full:
                self._capture_dropped += 1  # キューが満杯 = フレームドロップ (ディスクが遅い場合)

    def _capture_write_loop(self):
        """録画書き込みスレッド: キューからフレームを取り出してディスクに書き込む"""
        while self._capture_write_running:
            try:
                frame = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.recorder.write_frame(frame)
            except Exception as e:
                print(f"[CaptureWrite] エラー: {e}")
            finally:
                self._capture_queue.task_done()

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
                        buf_start = self.recorder.buffer_start
                        buf_cnt = self.recorder.buffered_frame_count
                        if buf_cnt > 0:
                            self.clip_slider.configure(
                                from_=buf_start,
                                to=max(buf_cnt - 1, buf_start),
                                command=None)
                            self.clip_slider.set(buf_cnt - 1)
                            self.clip_slider.configure(command=self._on_clip_slider)
                            self._clip_slider_frame = buf_cnt - 1
                            self._show_growing_preview(buf_cnt - 1)

            self._update_capture_timer_id = self.after(33, update)

        update()

    @staticmethod
    def _predetect_hw_encoder():
        """HWエンコーダを事前検出 (結果はffmpeg_writerモジュールにキャッシュされる)"""
        try:
            from ffmpeg_writer import detect_hw_encoder
            enc = detect_hw_encoder()
            if enc:
                print(f"[Startup] HWエンコーダ検出: {enc}")
            else:
                print("[Startup] HWエンコーダなし、ソフトウェアエンコードを使用")
        except Exception:
            pass

    def _toggle_rec(self):
        """REC/STOP切り替え"""
        if self.recorder.is_recording:
            # グローウィング中にセットされたIn/Outを保持
            g_in = self.recorder.growing_in
            g_out = self.recorder.growing_out
            # 新規フレームのキュー投入を停止 (コールバック側で判定)
            self._rec_stopping = True
            # UI即時更新 (ブロック防止)
            self.rec_btn.configure(text="⏺ REC", fg_color="#8B0000")
            self.edit_rec_btn.configure(text="⏺ REC", fg_color="#8B0000")
            self.po_rec_btn.configure(text="⏺ REC", fg_color="#8B0000")
            self.rec_status.configure(text="停止中...", text_color="yellow")

            def do_stop():
                # キュー残りを書き出し → ffmpeg終了 (バックグラウンドで実行)
                try:
                    self._capture_queue.join()
                except Exception:
                    pass
                self._rec_stopping = False
                frames = self.recorder.frame_count
                enqueued = getattr(self, '_capture_enqueued', 0)
                dropped = self._capture_dropped
                remaining = self._capture_queue.qsize()
                print(f"[REC DIAG] enqueued={enqueued}, written={frames}, "
                      f"dropped={dropped}, queue_remaining={remaining}")
                path = self.recorder.stop_recording()
                # GUIスレッドに戻して後処理
                self.after(0, lambda: self._finish_rec_stop(
                    path, frames, dropped, g_in, g_out))

            threading.Thread(target=do_stop, daemon=True,
                             name="RecStopThread").start()
        else:
            if not self.deck_input:
                self._start_capture()
            # 録画fps: 通常=29.97fps(等倍再生), HFR=29.97fps(スロー)
            rec_fps = self.deck_input.recording_fps if self.deck_input else self.settings["fps"]
            self._rec_frame_count = 0  # フレーム間引きカウンタリセット
            self._capture_enqueued = 0  # エンキューカウンタ
            self._capture_dropped = 0  # ドロップカウンタリセット
            if self.deck_input:
                cb_fps = getattr(self.deck_input, '_decklink', self.deck_input)
                mfps = getattr(cb_fps, '_measured_callback_fps', None)
                efps = getattr(self.deck_input, 'effective_fps', None)
                print(f"[REC] callback_fps={mfps}, effective_fps={efps}, recording_fps={rec_fps}")
            self.recorder.start_recording(fps=rec_fps)
            fps_label = f" ({rec_fps:.2f}fps)" if rec_fps != self.settings["fps"] else ""
            self.rec_btn.configure(text="⏹ STOP", fg_color="#FF0000")
            self.edit_rec_btn.configure(text="⏹ STOP", fg_color="#FF0000")
            self.po_rec_btn.configure(text="⏹ STOP", fg_color="#FF0000")
            self.rec_status.configure(text=f"● REC{fps_label}", text_color="red")
            self._clips_tab_dirty = self._edit_tab_dirty = True
            self._refresh_clips_list(scan=False)

    def _finish_rec_stop(self, path, frames, dropped, g_in, g_out):
        """録画停止の後処理 (GUIスレッドで実行)"""
        self.rec_status.configure(text="STANDBY", text_color="white")
        if dropped > 0:
            print(f"[Capture] 録画中にドロップしたフレーム数: {dropped}")
        if path and frames > 0:
            self.after(500, lambda p=str(path), gi=g_in, go=g_out:
                       self._add_recorded_clip(p, growing_in=gi, growing_out=go))
            # グローウィング切り出しクリップの高品質再書き出し
            self._start_growing_reexport()
        elif path and frames == 0:
            print(f"[Capture] 0フレーム録画のためスキップ: {path}")
        self._clips_tab_dirty = self._edit_tab_dirty = True
        self._refresh_clips_list(scan=False)

    def _start_growing_reexport(self):
        """グローウィング切り出しクリップをREC本体から高品質再書き出し (バックグラウンド)"""
        queue = self._growing_reexport_queue[:]
        self._growing_reexport_queue.clear()
        if not queue:
            return
        print(f"[Capture] グローウィング再書き出し: {len(queue)} 件")

        def do_reexport():
            from recorder import Recorder
            for rec_path, in_f, out_f, clip_path, fps in queue:
                ok = Recorder.re_export_clip(rec_path, in_f, out_f, clip_path, fps,
                                             crf=self.settings["crf"],
                                             hw_encode=True)
                if ok:
                    print(f"[Capture] 高品質差し替え完了: {Path(clip_path).name}")
                else:
                    print(f"[Capture] 高品質差し替え失敗: {Path(clip_path).name}")

        threading.Thread(target=do_reexport, daemon=True, name="GrowingReexport").start()

    def _add_recorded_clip(self, path, retries=3, growing_in=0, growing_out=-1):
        """録画ファイルをクリップに追加 (リトライ付き、グローウィングIn/Out引き継ぎ)"""
        try:
            # フォルダスキャンで既に登録済みなら再追加しない
            norm = os.path.normpath(path)
            clip = next((c for c in self.clip_manager.clips
                         if os.path.normpath(c.source_path) == norm), None)
            if clip is None:
                clip = self.clip_manager.add_clip(path)
            # グローウィング中に設定されたIn/Outを引き継ぎ
            if growing_in > 0 or growing_out >= 0:
                in_f = growing_in
                out_f = growing_out if growing_out >= 0 else clip.total_frames - 1
                self.clip_manager.set_in_out(clip.id, in_f, out_f)
                print(f"[Capture] グローウィングIn/Out引き継ぎ: {in_f}-{out_f}")
            self._clips_tab_dirty = self._edit_tab_dirty = True
            self._refresh_clips_list(scan=False)
            # 編集タブ表示中なら即座にクリップリストを更新
            if self.tabview.get() == "編集":
                self._refresh_edit_clips_list()
                self._edit_tab_dirty = False
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
                       command=lambda: self._refresh_clips_list(scan=True)).pack(side="left", padx=5)

        # クリップリスト（スクロール可能）
        self.clips_scroll = ctk.CTkScrollableFrame(list_frame)
        self.clips_scroll.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.clips_scroll.grid_columnconfigure(0, weight=1)
        self.clip_widgets = []
        self._collapsed_clip_groups = set()    # 折りたたみ中の日付グループ
        self._collapsed_playout_groups = set() # 送出リスト折りたたみ
        self._collapsed_edit_groups = set()   # 編集タブ折りたたみ
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

        # グローウィング更新ボタン (最新フレームに追従再開)
        self.growing_refresh_btn = ctk.CTkButton(
            right, text="⟳ 最新に更新", width=150, height=30,
            font=("", 12),
            fg_color="#2E4057", hover_color="#3D5A80",
            command=self._refresh_growing,
        )
        self.growing_refresh_btn.pack(padx=10, pady=(2, 0))
        self.growing_refresh_btn.pack_forget()  # 初期非表示

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

        self._refresh_clips_list(scan=True)

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
        today_dir = (self.clip_manager.clips_dir
                     / datetime.date.today().strftime("%m-%d"))
        today_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            src = Path(p)
            dest = today_dir / src.name
            # 重複回避
            if dest.exists():
                stem, suffix = src.stem, src.suffix
                i = 1
                while dest.exists():
                    dest = today_dir / f"{stem}_{i}{suffix}"
                    i += 1
            shutil.copy2(str(src), str(dest))
            self.clip_manager.add_clip(str(dest))
        self._clips_tab_dirty = self._edit_tab_dirty = True
        self._refresh_clips_list()

    def _sync_folder_clips(self):
        """recordings/ と clips/ フォルダの実ファイルからクリップリストを構築。
        - 実ファイルが無いエントリは除去
        - 同一ファイルの重複エントリは除去
        - フォルダにあって未登録のファイルは追加
        Path.resolve() を避け os.path.normpath で高速化。
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

        # 2) 同一source_pathの重複を除去 (normpath で比較)
        seen = set()
        deduped = []
        for c in self.clip_manager.clips:
            key = os.path.normpath(c.source_path)
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
        known_paths = {os.path.normpath(c.source_path) for c in self.clip_manager.clips}
        added = 0
        for d in scan_dirs:
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    key = os.path.normpath(str(f))
                    if key not in known_paths:
                        try:
                            self.clip_manager.add_clip(str(f))
                            known_paths.add(key)
                            added += 1
                        except Exception as e:
                            print(f"[Clips] スキャン追加エラー: {f.name}: {e}")
        if added:
            print(f"[Clips] フォルダスキャンで {added} 件追加")

    def _refresh_clips_list(self, scan=False):
        self._clips_tab_dirty = False
        if scan:
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

        # --- 保存済みクリップ (日付グループ表示 / 折りたたみ対応) ---
        def _date_key(clip):
            parent = Path(clip.source_path).parent.name
            if (len(parent) == 5 and parent[2] == '-') or (len(parent) == 10 and parent[4] == '-' and parent[7] == '-'):
                return parent
            return ""

        # 日付の新しい順 → "その他" は末尾
        sorted_clips = sorted(self.clip_manager.clips,
                              key=lambda c: _date_key(c) or "0000", reverse=True)
        current_group = None
        collapsed = False
        clip_num = 0
        for clip in sorted_clips:
            group = _date_key(clip)
            if group != current_group:
                current_group = group
                collapsed = group in self._collapsed_clip_groups
                arrow = "▶" if collapsed else "▼"
                label = group if group else "その他"
                header = ctk.CTkFrame(self.clips_scroll, height=25,
                                      fg_color="#1a3a1a", cursor="hand2")
                header.grid(row=row_idx, column=0, sticky="ew", pady=(6, 1))
                header.grid_columnconfigure(0, weight=1)
                hdr_btn = ctk.CTkButton(
                    header, text=f"{arrow} {label}",
                    font=("", 11, "bold"), text_color="#88CC88",
                    fg_color="transparent", hover_color="#2a4a2a",
                    anchor="w",
                    command=lambda g=group: self._toggle_clip_group(g))
                hdr_btn.grid(row=0, column=0, sticky="ew", padx=4)
                # 右クリック → フォルダを開く
                dir_path = str(Path(clip.source_path).parent)
                hdr_btn.bind("<Button-3>",
                             lambda e, d=dir_path: self._show_folder_menu(e, d))
                self.clip_widgets.append(header)
                row_idx += 1

            if collapsed:
                continue

            clip_num += 1
            row = ctk.CTkFrame(self.clips_scroll, height=40)
            row.grid(row=row_idx, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(row, text=f"{clip_num}", width=30).grid(row=0, column=0, padx=5)
            name_btn = ctk.CTkButton(
                row, text=clip.name, anchor="w",
                fg_color="transparent", hover_color="#333333",
                command=lambda cid=clip.id: self._select_clip(cid)
            )
            name_btn.grid(row=0, column=1, sticky="ew", padx=5)
            dur = f"{clip.duration_sec:.1f}s"
            traj = "✓" if clip.has_trajectory else ""
            ctk.CTkLabel(row, text=f"{dur}  {traj}", width=100).grid(row=0, column=2, padx=5)

            ctk.CTkButton(
                row, text="×", width=28, height=28,
                fg_color="#8B0000", hover_color="#A52A2A",
                font=("", 14, "bold"),
                command=lambda cid=clip.id: self._delete_clip_by_id(cid),
            ).grid(row=0, column=3, padx=(0, 5))

            self.clip_widgets.append(row)
            row_idx += 1

    def _toggle_clip_group(self, group):
        """クリップリストの日付グループ折りたたみ切替"""
        if group in self._collapsed_clip_groups:
            self._collapsed_clip_groups.discard(group)
        else:
            self._collapsed_clip_groups.add(group)
        self._refresh_clips_list()

    def _toggle_playout_group(self, group):
        """送出リストの日付グループ折りたたみ切替"""
        if group in self._collapsed_playout_groups:
            self._collapsed_playout_groups.discard(group)
        else:
            self._collapsed_playout_groups.add(group)
        self._refresh_playout_list()

    def _toggle_edit_group(self, group):
        """編集タブの日付グループ折りたたみ切替"""
        if group in self._collapsed_edit_groups:
            self._collapsed_edit_groups.discard(group)
        else:
            self._collapsed_edit_groups.add(group)
        self._refresh_edit_clips_list()

    def _show_folder_menu(self, event, dir_path):
        """右クリックでフォルダを開くコンテキストメニュー"""
        menu = tkinter.Menu(self, tearoff=0)
        menu.add_command(label="フォルダを開く",
                         command=lambda: os.startfile(dir_path))
        menu.tk_popup(event.x_root, event.y_root)

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

        self.clip_slider.configure(from_=0, to=max(clip.total_frames - 1, 1))
        self.clip_slider.set(clip.in_frame)
        self._show_clip_preview(clip, clip.in_frame)

        # グローウィング更新ボタンを非表示、切り出しボタンを表示
        self.growing_refresh_btn.pack_forget()
        self.clip_extract_btn.configure(text="クリップ切り出し", state="normal")
        self.clip_extract_btn.pack(padx=10, pady=(5, 0))

    def _release_clip_preview_cap(self):
        cap = getattr(self, '_clip_preview_cap', None)
        if cap:
            cap.release()
        self._clip_preview_cap = None
        self._clip_preview_cap_path = None

    def _show_clip_preview(self, clip, frame_no):
        # VideoCapture を使い回す (同じソースなら再オープンしない)
        if (getattr(self, '_clip_preview_cap_path', None) != clip.source_path
                or not getattr(self, '_clip_preview_cap', None)
                or not self._clip_preview_cap.isOpened()):
            self._release_clip_preview_cap()
            self._clip_preview_cap = cv2.VideoCapture(clip.source_path)
            self._clip_preview_cap_path = clip.source_path
        cap = self._clip_preview_cap
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
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

        buf_start = self.recorder.buffer_start
        slider_max = max(buf_cnt - 1, buf_start)
        self.clip_slider.configure(from_=buf_start, to=slider_max, command=None)
        self.clip_slider.set(min(max(buf_cnt - 1, buf_start), slider_max))
        self.clip_slider.configure(command=self._on_clip_slider)
        self._clip_slider_frame = min(max(buf_cnt - 1, buf_start), slider_max)
        if buf_cnt > 0:
            self._show_growing_preview(buf_cnt - 1)

        # グローウィング用ボタン表示
        self.growing_refresh_btn.pack(padx=10, pady=(2, 0))
        self.clip_extract_btn.pack(padx=10, pady=(5, 0))

    def _refresh_growing(self):
        """グローウィングのスライダー最大値を更新し、最新フレームに追従再開"""
        if self._selected_clip_id != "__growing__":
            return
        buf_start = self.recorder.buffer_start
        buf_cnt = self.recorder.buffered_frame_count
        if buf_cnt <= 0:
            return
        slider_max = max(buf_cnt - 1, buf_start)
        self.clip_slider.configure(from_=buf_start, to=slider_max, command=None)
        self.clip_slider.set(buf_cnt - 1)
        self.clip_slider.configure(command=self._on_clip_slider)
        self._clip_slider_frame = buf_cnt - 1
        self._growing_follow_live = True
        self._show_growing_preview(buf_cnt - 1)

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
            min_f = self.recorder.buffer_start
            max_f = max(self.recorder.buffered_frame_count - 1, min_f)
        else:
            clip = self.clip_manager.get_clip(self._selected_clip_id)
            if not clip:
                return
            min_f = 0
            max_f = max(clip.total_frames - 1, 0)
        new_f = max(min_f, min(new_f, max_f))
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
        path = self.clip_manager.export_trimmed(self._selected_clip_id,
                                                 crf=self.settings["crf"],
                                                 hw_encode=True)
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
            ts = time.strftime("%m%d_%H%M%S")
            clip_name = f"clip_{ts}_{in_f}_{out_f}"
            out_dir = self.project_dir / "clips" / datetime.date.today().strftime("%m-%d")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{clip_name}.mp4"

            rec_path = str(self.recorder.current_file)
            rec_fps = self.recorder.fps

            def do_export():
                count = self.recorder.export_clip(in_f, out_f, str(out_path))
                if count > 0:
                    # REC停止後に高品質再書き出しするためキューに追加
                    self._growing_reexport_queue.append(
                        (rec_path, in_f, out_f, str(out_path), rec_fps))
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
            path = self.clip_manager.export_trimmed(clip_id,
                                                     crf=self.settings["crf"],
                                                     hw_encode=True)
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
            self._clips_tab_dirty = self._edit_tab_dirty = True
            self._refresh_clips_list(scan=False)
            # 編集タブ表示中なら即座にクリップリストを更新
            if self.tabview.get() == "編集":
                self._refresh_edit_clips_list()
                self._edit_tab_dirty = False
            # 新しいクリップを選択
            self._select_clip(clip.id)
            print(f"[Capture] グローウィングクリップ追加: {clip.name}")
        except Exception as e:
            print(f"[Capture] クリップ追加エラー: {e}")
        finally:
            self.clip_extract_btn.configure(text="クリップ切り出し", state="normal")

    def _delete_clip_files(self, clip_id):
        """クリップに紐づく実ファイル (source, exported, trajectory) を削除
        他のクリップが source_path として参照しているファイルは保護する。"""
        clip = self.clip_manager.get_clip(clip_id)
        if not clip:
            return
        # 他クリップが source_path として使用中のパスを収集
        protected = set()
        for c in self.clip_manager.clips:
            if c.id != clip_id and c.source_path:
                protected.add(str(Path(c.source_path).resolve()))
        for p in [clip.source_path, clip.exported_path, clip.trajectory_path]:
            if p:
                fp = Path(p)
                if str(fp.resolve()) in protected:
                    print(f"[Clips] 他クリップが参照中のため保護: {fp.name}")
                    continue
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
            self._clips_tab_dirty = self._edit_tab_dirty = True
            self._refresh_clips_list()

    def _delete_clip_by_id(self, clip_id):
        """クリップリストの行ボタンから直接削除"""
        self._delete_clip_files(clip_id)
        self.clip_manager.remove_clip(clip_id)
        if self._selected_clip_id == clip_id:
            self._selected_clip_id = None
        self._clips_tab_dirty = self._edit_tab_dirty = True
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
        self.edit_canvas.bind("<Button-2>", self._edit_middle_press)
        self.edit_canvas.bind("<B2-Motion>", self._edit_middle_drag)
        self.edit_canvas.bind("<ButtonRelease-2>", self._edit_middle_release)
        self.edit_canvas.bind("<Button-3>", self._edit_right_press)
        self.edit_canvas.bind("<B3-Motion>", self._edit_right_drag)
        self.edit_canvas.bind("<ButtonRelease-3>", self._edit_right_release)
        self.edit_canvas.bind("<MouseWheel>", self._edit_on_wheel)
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

        # IN/OUT マーカー (▲をドラッグ or ショートカットキーで設定)
        # シークバーと同じカラム (column=1) に配置して位置を揃える
        self._edit_in = 0
        self._edit_out = 0
        self._io_canvas = Canvas(tl_frame, height=28, bg="#2b2b2b", highlightthickness=0)
        self._io_canvas.grid(row=1, column=1, sticky="ew", pady=(0, 0))
        self._io_canvas.bind("<Configure>", self._io_redraw)
        self._io_canvas.bind("<Button-1>", self._io_press)
        self._io_canvas.bind("<B1-Motion>", self._io_drag)
        self._io_dragging = None  # "in" or "out"

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

        # --- RECボタン + クリップリスト (上端固定) ---
        self.edit_rec_btn = ctk.CTkButton(
            right, text="⏺ REC", width=250, height=36,
            font=("", 14, "bold"),
            fg_color="#8B0000", hover_color="#B22222",
            command=self._toggle_rec
        )
        self.edit_rec_btn.pack(padx=10, pady=(8, 2))
        ctk.CTkLabel(right, text="クリップ", font=("", 14, "bold")).pack(pady=(4, 2))
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
        self._edit_spline_cache = {}  # swing_idx → (key, TimedSpline)
        self._edit_swing_idx = 0
        self._edit_playing = False
        self._edit_slider_updating = False  # スライダー循環呼び出し防止
        self._edit_slider_after = None       # スライダーデバウンス用 after ID
        self._edit_scale = 1.0
        self._edit_zoom = 1.0             # ユーザーズーム倍率 (1.0=フィット)
        self._edit_pan_vx = 0.0           # パンオフセット (動画座標)
        self._edit_pan_vy = 0.0
        self._edit_mid_press = None       # 中ボタンドラッグ開始位置
        self._edit_mid_moved = False
        self._edit_dragging = None
        self._edit_right_press_pos = None  # 右クリック開始位置 (クリックvs.ドラッグ判定用)
        self._edit_right_moved = False     # 右ドラッグ中にマウスが動いたか
        self._edit_handle_point = None     # ハンドル編集中のポイント (swing_idx, point_idx)
        self._edit_dragging_handle = None  # ドラッグ中のハンドル ('in' or 'out', swing_idx, point_idx)
        self._edit_undo_stack = []         # Undo スナップショットスタック
        self._edit_redo_stack = []         # Redo スナップショットスタック
        self._UNDO_MAX = 50

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

        def _date_key(clip):
            parent = Path(clip.source_path).parent.name
            if (len(parent) == 5 and parent[2] == '-') or (len(parent) == 10 and parent[4] == '-' and parent[7] == '-'):
                return parent
            return ""

        sorted_clips = sorted(self.clip_manager.clips,
                              key=lambda c: _date_key(c) or "0000", reverse=True)
        current_group = None
        collapsed = False
        row_idx = 0
        for clip in sorted_clips:
            group = _date_key(clip)
            if group != current_group:
                current_group = group
                collapsed = group in self._collapsed_edit_groups
                arrow = "▶" if collapsed else "▼"
                label = group if group else "その他"
                header = ctk.CTkFrame(self.edit_clips_scroll, height=22,
                                      fg_color="#1a3a1a", cursor="hand2")
                header.grid(row=row_idx, column=0, sticky="ew", pady=(4, 1))
                header.grid_columnconfigure(0, weight=1)
                hdr_btn = ctk.CTkButton(
                    header, text=f"{arrow} {label}",
                    font=("", 11, "bold"), text_color="#88CC88",
                    fg_color="transparent", hover_color="#2a4a2a",
                    anchor="w",
                    command=lambda g=group: self._toggle_edit_group(g))
                hdr_btn.grid(row=0, column=0, sticky="ew", padx=4)
                dir_path = str(Path(clip.source_path).parent)
                hdr_btn.bind("<Button-3>",
                             lambda e, d=dir_path: self._show_folder_menu(e, d))
                self._edit_clip_widgets.append(header)
                row_idx += 1

            if collapsed:
                continue

            row = ctk.CTkFrame(self.edit_clips_scroll, height=30)
            row.grid(row=row_idx, column=0, sticky="ew", pady=1)
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
            row_idx += 1

    def _edit_select_clip(self, clip_id):
        """編集タブでクリップを選択して軌道編集にロード"""
        # 現在編集中の軌道を先に自動保存
        if self._edit_clip_id and self._edit_clip_id != clip_id:
            self._edit_autosave_trajectory()
        self._edit_clip_id = clip_id
        self._edit_undo_stack.clear()
        self._edit_redo_stack.clear()
        self._load_edit_clip()
        self._refresh_edit_clips_list()

    @property
    def _edit_current_swing(self):
        if self._edit_swings and 0 <= self._edit_swing_idx < len(self._edit_swings):
            return self._edit_swings[self._edit_swing_idx]
        return None

    def _make_default_trajectory(self, color_idx=0):
        """最後に使ったスタイルで新規 TrajectoryData を作成
        Swing 1 は保存済みスタイルの色を使用、Swing 2以降はプリセット"""
        style = self.settings.data.get("trajectory_style", {})
        if color_idx == 0 and "color_start_hex" in style:
            c_start = style["color_start_hex"]
            c_end = style["color_end_hex"]
        else:
            preset = GRADIENT_PRESETS[color_idx % len(GRADIENT_PRESETS)]
            c_start, c_end = preset[0], preset[1]
        return TrajectoryData(
            color_start_hex=c_start,
            color_end_hex=c_end,
            thickness=style.get("thickness", 3),
            blur=style.get("blur", 0),
            fade_frames=style.get("fade_frames", 0),
            alpha=style.get("alpha", 0.85),
        )

    def _save_trajectory_style(self, swing):
        """現在のスイング設定を次回のデフォルトとして保存"""
        if not swing:
            return
        self.settings["trajectory_style"] = {
            "color_start_hex": swing.color_start_hex,
            "color_end_hex": swing.color_end_hex,
            "thickness": swing.thickness,
            "blur": getattr(swing, "blur", 0),
            "fade_frames": getattr(swing, "fade_frames", 0),
            "alpha": getattr(swing, "alpha", 0.85),
        }
        self.settings.save()

    def _load_edit_clip(self):
        """編集タブにクリップをロード (最初のフレームを即表示、残りはバックグラウンド)"""
        clip = self.clip_manager.get_clip(self._edit_clip_id)
        if not clip:
            return

        print(f"[Edit] クリップ読み込み: {clip.name}")
        # スライダーデバウンスをキャンセル (古いクリップの更新が残らないように)
        if self._edit_slider_after is not None:
            self.after_cancel(self._edit_slider_after)
            self._edit_slider_after = None
        self._edit_direct_clip = None
        self._edit_cache = FrameCache(clip.source_path, clip.in_frame, clip.get_out_frame())
        self._edit_total = len(self._edit_cache)
        self._edit_frame_no = 0
        self._edit_zoom = 1.0
        self._edit_pan_vx = 0.0
        self._edit_pan_vy = 0.0

        # 既存軌道を読み込み
        saved = self.clip_manager.load_trajectory(self._edit_clip_id)
        if saved:
            self._edit_swings = saved
        else:
            self._edit_swings = [self._make_default_trajectory(0)]
        self._edit_spline_cache.clear()
        self._edit_swing_idx = 0

        self.edit_slider.configure(to=max(self._edit_total - 1, 1))
        self._edit_reset_in_out()
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

        # --- 先にクロップ＆リサイズ (表示サイズに縮小) ---
        cw = self.edit_canvas.winfo_width()
        ch = self.edit_canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 450

        fh, fw = frame.shape[:2]
        self._edit_scale = min(cw / fw, ch / fh) if fw > 0 and fh > 0 else 1.0
        eff_scale = self._edit_scale * self._edit_zoom

        if self._edit_zoom == 1.0:
            self._edit_pan_vx = fw / 2
            self._edit_pan_vy = fh / 2

        view_w = cw / eff_scale
        view_h = ch / eff_scale
        vx0 = self._edit_pan_vx - view_w / 2
        vy0 = self._edit_pan_vy - view_h / 2
        vx1 = vx0 + view_w
        vy1 = vy0 + view_h

        src_x0 = max(0, int(vx0))
        src_y0 = max(0, int(vy0))
        src_x1 = min(fw, int(np.ceil(vx1)))
        src_y1 = min(fh, int(np.ceil(vy1)))

        cropped = frame[src_y0:src_y1, src_x0:src_x1]
        if cropped.size == 0:
            cropped = frame
            src_x0, src_y0 = 0, 0

        disp_w = max(1, int((src_x1 - src_x0) * eff_scale))
        disp_h = max(1, int((src_y1 - src_y0) * eff_scale))
        interp = cv2.INTER_AREA if eff_scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(cropped, (disp_w, disp_h), interpolation=interp)

        # --- 軌道描画 (リサイズ後の小さい画像上 → 高速) ---
        cur = self._edit_frame_no
        for swing in self._edit_swings:
            if len(swing.points) < 2:
                if swing.points:
                    scaled_pts = [
                        (int((p[0] - src_x0) * eff_scale),
                         int((p[1] - src_y0) * eff_scale), p[2])
                        for p in swing.points if p[2] <= cur]
                    if scaled_pts:
                        draw_markers(resized, scaled_pts,
                                     hex_to_bgr(swing.color_start_hex),
                                     hex_to_bgr(swing.color_end_hex),
                                     max(1, int(MARKER_RADIUS * eff_scale)))
                continue

            blur = getattr(swing, 'blur', 0)
            base_a = getattr(swing, 'alpha', 0.85)
            eff_alpha = base_a

            handles = getattr(swing, 'handles', [])
            si = self._edit_swings.index(swing)
            cache_key = (tuple((p[0], p[1], p[2]) for p in swing.points),
                         tuple(h if h is None else (h[0], h[1])
                               for h in handles) if handles else ())
            cached = self._edit_spline_cache.get(si)
            if cached and cached[0] == cache_key:
                ts = cached[1]
            else:
                ts = TimedSpline(swing.points, SPLINE_RESOLUTION,
                                 handles=handles if handles else None)
                self._edit_spline_cache[si] = (cache_key, ts)
            curve_pts = ts.get_curve_at_frame(cur)
            if curve_pts and len(curve_pts) >= 2:
                # 曲線座標を表示座標に変換
                scaled_curve = [
                    (int((p[0] - src_x0) * eff_scale),
                     int((p[1] - src_y0) * eff_scale))
                    for p in curve_pts]
                c_start = hex_to_bgr(swing.color_start_hex)
                c_end = hex_to_bgr(swing.color_end_hex)
                full_len = len(ts._curve)
                ratio = len(curve_pts) / max(full_len, 1)
                c_end_anim = lerp_color_bgr(c_start, c_end, ratio)
                scaled_thick = max(1, int(swing.thickness * eff_scale))
                scaled_blur = int(blur * eff_scale) if blur > 0 else 0
                draw_gradient_trail(resized, scaled_curve, c_start,
                                    c_end_anim, scaled_thick,
                                    eff_alpha, blur=scaled_blur)
            if swing.points:
                visible_pts = [
                    (int((p[0] - src_x0) * eff_scale),
                     int((p[1] - src_y0) * eff_scale), p[2])
                    for p in swing.points if p[2] <= cur]
                if visible_pts:
                    draw_markers(resized, visible_pts,
                                 hex_to_bgr(swing.color_start_hex),
                                 hex_to_bgr(swing.color_end_hex),
                                 max(1, int(MARKER_RADIUS * eff_scale)))

        img_cx = int(cw / 2 - (self._edit_pan_vx - (src_x0 + src_x1) / 2) * eff_scale)
        img_cy = int(ch / 2 - (self._edit_pan_vy - (src_y0 + src_y1) / 2) * eff_scale)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._edit_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.edit_canvas.delete("all")
        self.edit_canvas.create_image(img_cx, img_cy, anchor="center",
                                       image=self._edit_photo)

        # ベジェハンドル描画 (キャンバス上にオーバーレイ)
        for si, swing in enumerate(self._edit_swings):
            handles = getattr(swing, 'handles', [])
            for pi, pt in enumerate(swing.points):
                if pi >= len(handles) or handles[pi] is None:
                    continue
                h = handles[pi]
                pcx, pcy = self._edit_video_to_canvas(pt[0], pt[1])
                # 入射ハンドル
                hx_in = pt[0] + h[0]
                hy_in = pt[1] + h[1]
                cx_in, cy_in = self._edit_video_to_canvas(hx_in, hy_in)
                # 出射ハンドル
                hx_out = pt[0] + h[2]
                hy_out = pt[1] + h[3]
                cx_out, cy_out = self._edit_video_to_canvas(hx_out, hy_out)
                # ハンドル線
                self.edit_canvas.create_line(
                    pcx, pcy, cx_in, cy_in, fill="#00BFFF", width=1, dash=(4, 2))
                self.edit_canvas.create_line(
                    pcx, pcy, cx_out, cy_out, fill="#FF6347", width=1, dash=(4, 2))
                # ハンドル点 (□)
                hr = 4
                active = self._edit_handle_point == (si, pi)
                fill_in = "#00BFFF" if active else "#005f7f"
                fill_out = "#FF6347" if active else "#7f3123"
                self.edit_canvas.create_rectangle(
                    cx_in - hr, cy_in - hr, cx_in + hr, cy_in + hr,
                    fill=fill_in, outline="white", width=1)
                self.edit_canvas.create_rectangle(
                    cx_out - hr, cy_out - hr, cx_out + hr, cy_out + hr,
                    fill=fill_out, outline="white", width=1)

        zoom_txt = f" ({self._edit_zoom:.0f}x)" if self._edit_zoom > 1.0 else ""
        self.edit_frame_label.configure(
            text=f"{self._edit_frame_no} / {self._edit_total - 1}{zoom_txt}")
        self._edit_slider_updating = True
        self.edit_slider.set(self._edit_frame_no)
        self._edit_slider_updating = False

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

        eff_scale = self._edit_scale * self._edit_zoom
        # キャンバス中心 = パンオフセットされた動画座標の中心
        vx = (cx - cw / 2) / eff_scale + self._edit_pan_vx
        vy = (cy - ch / 2) / eff_scale + self._edit_pan_vy
        return int(vx), int(vy)

    def _edit_left_click(self, event):
        swing = self._edit_current_swing
        if not swing:
            return
        self._edit_push_undo()
        vx, vy = self._edit_canvas_to_video(event.x, event.y)
        swing.points.append((vx, vy, self._edit_frame_no))
        # ハンドルも同期 (新規ポイントは None)
        self._edit_ensure_handles(swing)
        # points と handles をペアでフレーム順ソート
        paired = list(zip(swing.points, swing.handles))
        paired.sort(key=lambda x: x[0][2])
        swing.points[:] = [p for p, h in paired]
        swing.handles[:] = [h for p, h in paired]
        print(f"  Point: ({vx}, {vy}) @ frame {self._edit_frame_no}")
        self._edit_update_display()

    def _edit_middle_press(self, event):
        """中ボタン押下: パンドラッグ開始"""
        self._edit_mid_press = (event.x, event.y)
        self._edit_mid_moved = False
        self._edit_pan_start_vx = self._edit_pan_vx
        self._edit_pan_start_vy = self._edit_pan_vy
        self.edit_canvas.config(cursor="fleur")

    def _edit_middle_drag(self, event):
        """中ボタンドラッグ: パン"""
        if self._edit_mid_press is None:
            return
        dx = event.x - self._edit_mid_press[0]
        dy = event.y - self._edit_mid_press[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._edit_mid_moved = True
        eff_scale = self._edit_scale * self._edit_zoom
        self._edit_pan_vx = self._edit_pan_start_vx - dx / eff_scale
        self._edit_pan_vy = self._edit_pan_start_vy - dy / eff_scale
        self._edit_clamp_pan()
        self._edit_update_display()

    def _edit_middle_release(self, event):
        """中ボタンリリース: ドラッグなし→ポイント削除"""
        self.edit_canvas.config(cursor="")
        if not self._edit_mid_moved:
            # クリック: 最寄りの点を削除 (従来動作)
            vx, vy = self._edit_canvas_to_video(event.x, event.y)
            result = self._edit_find_nearest(vx, vy)
            eff = self._edit_scale * self._edit_zoom
            thresh = POINT_GRAB_RADIUS / max(eff, 0.1)
            if result and result[2] < thresh:
                si, pi = result[0], result[1]
                swing = self._edit_swings[si]
                self._edit_push_undo()
                removed = swing.points.pop(pi)
                if swing.handles and pi < len(swing.handles):
                    swing.handles.pop(pi)
                if self._edit_handle_point == (si, pi):
                    self._edit_handle_point = None
                print(f"  Point deleted: ({removed[0]}, {removed[1]}) @ frame {removed[2]}")
                self._edit_update_display()
        self._edit_mid_press = None

    def _edit_clamp_pan(self):
        """パン位置をクランプしてフレーム外が見えないようにする"""
        if not self._edit_cache or self._edit_total == 0:
            return
        frame = self._edit_cache[0]
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        cw = self.edit_canvas.winfo_width()
        ch = self.edit_canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 450
        eff_scale = self._edit_scale * self._edit_zoom
        view_w = cw / eff_scale
        view_h = ch / eff_scale
        # 可視範囲がフレームより大きい場合は中央固定
        if view_w >= fw:
            self._edit_pan_vx = fw / 2
        else:
            self._edit_pan_vx = max(view_w / 2, min(fw - view_w / 2, self._edit_pan_vx))
        if view_h >= fh:
            self._edit_pan_vy = fh / 2
        else:
            self._edit_pan_vy = max(view_h / 2, min(fh - view_h / 2, self._edit_pan_vy))

    def _edit_on_wheel(self, event):
        """マウスホイール: ズーム (カーソル中心)"""
        if not self._edit_cache or self._edit_total == 0:
            return
        # カーソル位置の動画座標 (float精度で計算 — int切り捨てによるズレを防ぐ)
        cw = self.edit_canvas.winfo_width()
        ch = self.edit_canvas.winfo_height()
        eff_scale = self._edit_scale * self._edit_zoom
        vx_f = (event.x - cw / 2) / eff_scale + self._edit_pan_vx
        vy_f = (event.y - ch / 2) / eff_scale + self._edit_pan_vy

        # ズーム倍率更新
        if event.delta > 0:
            self._edit_zoom = min(self._edit_zoom * 1.25, 20.0)
        else:
            self._edit_zoom = max(self._edit_zoom / 1.25, 1.0)

        if self._edit_zoom == 1.0:
            # フィットに戻す: パンリセット
            frame = self._edit_cache[0]
            if frame is not None:
                fh, fw = frame.shape[:2]
                self._edit_pan_vx = fw / 2
                self._edit_pan_vy = fh / 2
        else:
            # カーソル位置を固定してズーム (カーソル下の映像がずれない)
            new_eff = self._edit_scale * self._edit_zoom
            self._edit_pan_vx = vx_f - (event.x - cw / 2) / new_eff
            self._edit_pan_vy = vy_f - (event.y - ch / 2) / new_eff
        self._edit_clamp_pan()

        self._edit_update_display()

    def _edit_find_nearest(self, vx, vy):
        best = None
        for si, swing in enumerate(self._edit_swings):
            for pi, pt in enumerate(swing.points):
                d = np.hypot(vx - pt[0], vy - pt[1])
                if best is None or d < best[2]:
                    best = (si, pi, d)
        return best

    def _edit_video_to_canvas(self, vx, vy):
        """動画座標 → キャンバス座標"""
        cw = self.edit_canvas.winfo_width()
        ch = self.edit_canvas.winfo_height()
        if not self._edit_cache or self._edit_total == 0:
            return 0, 0
        eff_scale = self._edit_scale * self._edit_zoom
        cx = (vx - self._edit_pan_vx) * eff_scale + cw / 2
        cy = (vy - self._edit_pan_vy) * eff_scale + ch / 2
        return int(cx), int(cy)

    def _edit_find_nearest_handle(self, vx, vy):
        """ハンドル編集中のポイントのハンドルに近いか判定
        Returns: ('in'|'out', si, pi, dist) or None"""
        hp = self._edit_handle_point
        if hp is None:
            return None
        si, pi = hp
        swing = self._edit_swings[si]
        if pi >= len(swing.points):
            return None
        pt = swing.points[pi]
        handles = getattr(swing, 'handles', [])
        if pi >= len(handles) or handles[pi] is None:
            return None
        h = handles[pi]
        # 入射ハンドル
        hx_in, hy_in = pt[0] + h[0], pt[1] + h[1]
        d_in = np.hypot(vx - hx_in, vy - hy_in)
        # 出射ハンドル
        hx_out, hy_out = pt[0] + h[2], pt[1] + h[3]
        d_out = np.hypot(vx - hx_out, vy - hy_out)
        best_kind = 'in' if d_in < d_out else 'out'
        best_d = min(d_in, d_out)
        return (best_kind, si, pi, best_d)

    def _edit_ensure_handles(self, swing):
        """ハンドルリストをポイント数に合わせる"""
        while len(swing.handles) < len(swing.points):
            swing.handles.append(None)
        while len(swing.handles) > len(swing.points):
            swing.handles.pop()

    def _edit_init_handle(self, swing, pi):
        """ポイントにデフォルトハンドルを設定 (splprep接線ベース)
        既存のスプライン曲線の接線を流用し、曲線形状を保つ"""
        pts = swing.points
        n = len(pts)
        if n < 2:
            swing.handles[pi] = (0, 0, 0, 0)
            return

        # splprepの接線を計算 (ハンドルなしの状態でTimedSplineを構築)
        ts = TimedSpline(pts, SPLINE_RESOLUTION)
        tangents = ts._compute_spline_tangents(
            [p[0] for p in pts], [p[1] for p in pts],
            ts._build_u_pts()
            if hasattr(ts, '_build_u_pts') else self._calc_u_pts(pts))
        tx, ty = tangents[pi]

        # セグメント長比でスケール (ベジェ制御点 = 接線 * seg_frac/3)
        u_pts = self._calc_u_pts(pts)
        # piの前後セグメントの平均seg_fracを使用
        if pi == 0:
            seg_frac = u_pts[1] - u_pts[0]
        elif pi == n - 1:
            seg_frac = u_pts[-1] - u_pts[-2]
        else:
            seg_frac = (u_pts[pi + 1] - u_pts[pi - 1]) * 0.5
        scale = seg_frac / 3.0
        hx, hy = tx * scale, ty * scale
        # in-handle = 逆方向, out-handle = 正方向
        swing.handles[pi] = (int(-hx), int(-hy), int(hx), int(hy))

    @staticmethod
    def _calc_u_pts(pts):
        """ポイント列から距離ベースのuパラメータを計算"""
        n = len(pts)
        dists = [0.0]
        for i in range(1, n):
            d = np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
            dists.append(dists[-1] + max(d, 1e-6))
        total = dists[-1]
        return [d / total for d in dists]

    def _edit_right_press(self, event):
        self._edit_right_press_pos = (event.x, event.y)
        self._edit_right_moved = False
        vx, vy = self._edit_canvas_to_video(event.x, event.y)
        thresh = POINT_GRAB_RADIUS / max(self._edit_scale * self._edit_zoom, 0.1)

        # ハンドル編集中ならハンドルのドラッグを優先
        handle_result = self._edit_find_nearest_handle(vx, vy)
        if handle_result and handle_result[3] < thresh:
            kind, si, pi = handle_result[0], handle_result[1], handle_result[2]
            self._edit_push_undo()
            self._edit_dragging_handle = (kind, si, pi)
            self._edit_dragging = None
            self.edit_canvas.config(cursor="crosshair")
            return

        # ポイント移動
        result = self._edit_find_nearest(vx, vy)
        if result and result[2] < thresh:
            self._edit_push_undo()
            self._edit_dragging = (result[0], result[1])
            self._edit_dragging_handle = None
            self.edit_canvas.config(cursor="fleur")
        else:
            self._edit_dragging = None
            self._edit_dragging_handle = None

    def _edit_right_drag(self, event):
        self._edit_right_moved = True

        # ハンドルドラッグ
        if self._edit_dragging_handle is not None:
            kind, si, pi = self._edit_dragging_handle
            vx, vy = self._edit_canvas_to_video(event.x, event.y)
            swing = self._edit_swings[si]
            pt = swing.points[pi]
            dx, dy = vx - pt[0], vy - pt[1]
            old = swing.handles[pi]
            if kind == 'out':
                swing.handles[pi] = (int(-dx), int(-dy), int(dx), int(dy))
            else:
                swing.handles[pi] = (int(dx), int(dy), int(-dx), int(-dy))
            self._edit_update_display()
            return

        # ポイント移動 (従来動作)
        if self._edit_dragging is None:
            return
        si, pi = self._edit_dragging
        vx, vy = self._edit_canvas_to_video(event.x, event.y)
        old = self._edit_swings[si].points[pi]
        self._edit_swings[si].points[pi] = (vx, vy, old[2])
        self._edit_update_display()

    def _edit_right_release(self, event):
        was_dragging_handle = self._edit_dragging_handle is not None
        self._edit_dragging_handle = None

        if not self._edit_right_moved and not was_dragging_handle:
            # ドラッグなし = クリック → ハンドル表示/非表示トグル
            vx, vy = self._edit_canvas_to_video(event.x, event.y)
            result = self._edit_find_nearest(vx, vy)
            thresh = POINT_GRAB_RADIUS / max(self._edit_scale * self._edit_zoom, 0.1)
            if result and result[2] < thresh:
                si, pi = result[0], result[1]
                swing = self._edit_swings[si]
                self._edit_ensure_handles(swing)
                self._edit_push_undo()
                if swing.handles[pi] is not None:
                    # ハンドルOFF
                    swing.handles[pi] = None
                    if self._edit_handle_point == (si, pi):
                        self._edit_handle_point = None
                    print(f"  Handle OFF: point {pi}")
                else:
                    # ハンドルON
                    self._edit_init_handle(swing, pi)
                    self._edit_handle_point = (si, pi)
                    print(f"  Handle ON: point {pi}")
            else:
                # 空白クリック → ハンドル選択解除
                self._edit_handle_point = None

        self._edit_dragging = None
        self._edit_right_press_pos = None
        self.edit_canvas.config(cursor="")
        self._edit_update_display()

    def _edit_jump(self, delta):
        self._edit_frame_no = max(0, min(self._edit_frame_no + delta, self._edit_total - 1))
        self._edit_update_display()

    def _edit_set_in(self):
        """現在フレームを IN 点に設定"""
        self._edit_in = self._edit_frame_no
        if self._edit_in > self._edit_out:
            self._edit_out = self._edit_in
        self._io_redraw()

    def _edit_set_out(self):
        """現在フレームを OUT 点に設定"""
        self._edit_out = self._edit_frame_no
        if self._edit_out < self._edit_in:
            self._edit_in = self._edit_out
        self._io_redraw()

    def _edit_reset_in_out(self):
        """IN/OUT をデフォルト (全範囲) にリセット"""
        self._edit_in = 0
        self._edit_out = max(0, self._edit_total - 1)
        self._io_redraw()

    # --- IN/OUT マーカー Canvas ---
    # CTkSliderの内部トラック位置に合わせるパディング
    _IO_PAD = 8

    def _io_frame_to_x(self, frame):
        """フレーム番号 → Canvas x座標"""
        cw = self._io_canvas.winfo_width()
        pad = self._IO_PAD
        total = max(self._edit_total - 1, 1)
        return pad + (cw - 2 * pad) * frame / total

    def _io_x_to_frame(self, x):
        """Canvas x座標 → フレーム番号"""
        cw = self._io_canvas.winfo_width()
        pad = self._IO_PAD
        total = max(self._edit_total - 1, 1)
        f = int(round((x - pad) / max(cw - 2 * pad, 1) * total))
        return max(0, min(f, self._edit_total - 1))

    def _io_redraw(self, event=None):
        """IN/OUT マーカーを再描画"""
        c = self._io_canvas
        c.delete("all")
        cw = c.winfo_width()
        if cw < 20 or self._edit_total <= 0:
            return
        # バー背景
        y_bar = 4
        pad = self._IO_PAD
        c.create_line(pad, y_bar, cw - pad, y_bar, fill="#555", width=1)
        # IN-OUT 範囲ハイライト
        x_in = self._io_frame_to_x(self._edit_in)
        x_out = self._io_frame_to_x(self._edit_out)
        c.create_line(x_in, y_bar, x_out, y_bar, fill="#4a9eff", width=3)
        # IN マーカー ▲
        c.create_polygon(x_in, y_bar + 2, x_in - 5, y_bar + 12, x_in + 5, y_bar + 12,
                         fill="#4a9eff", outline="#fff", tags="in_marker")
        c.create_text(x_in, y_bar + 15, text=f"IN:{self._edit_in}", fill="#aaa",
                       font=("", 8), anchor="n")
        # OUT マーカー ▲
        c.create_polygon(x_out, y_bar + 2, x_out - 5, y_bar + 12, x_out + 5, y_bar + 12,
                         fill="#ff6a4a", outline="#fff", tags="out_marker")
        c.create_text(x_out, y_bar + 15, text=f"OUT:{self._edit_out}", fill="#aaa",
                       font=("", 8), anchor="n")

    def _io_press(self, event):
        """マーカーのドラッグ開始: 近い方のマーカーを掴む"""
        x_in = self._io_frame_to_x(self._edit_in)
        x_out = self._io_frame_to_x(self._edit_out)
        d_in = abs(event.x - x_in)
        d_out = abs(event.x - x_out)
        if d_in <= d_out and d_in < 30:
            self._io_dragging = "in"
        elif d_out < 30:
            self._io_dragging = "out"
        else:
            self._io_dragging = None

    def _io_drag(self, event):
        """マーカーのドラッグ中"""
        if not self._io_dragging:
            return
        frame = self._io_x_to_frame(event.x)
        if self._io_dragging == "in":
            self._edit_in = min(frame, self._edit_out)
        else:
            self._edit_out = max(frame, self._edit_in)
        self._io_redraw()

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
        if self._edit_slider_updating:
            return
        self._edit_frame_no = int(value)
        # フレーム番号ラベルを即座に更新 (応答性向上)
        zoom_txt = f" ({self._edit_zoom:.0f}x)" if self._edit_zoom > 1.0 else ""
        self.edit_frame_label.configure(
            text=f"{self._edit_frame_no} / {self._edit_total - 1}{zoom_txt}")
        # デバウンス: 高速ドラッグ中は最後の値だけ描画 (~30fps上限)
        if self._edit_slider_after is not None:
            self.after_cancel(self._edit_slider_after)
        self._edit_slider_after = self.after(30, self._edit_slider_flush)

    def _edit_slider_flush(self):
        """デバウンス後に実際の表示更新を実行"""
        self._edit_slider_after = None
        self._edit_update_display()

    def _edit_toggle_play(self):
        self._edit_playing = not self._edit_playing
        self.edit_play_btn.configure(text="⏸" if self._edit_playing else "▶")
        if self._edit_playing:
            self._edit_play_loop()

    def _edit_play_loop(self):
        if not self._edit_playing:
            return
        if self._edit_frame_no >= self._edit_out:
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
            self._save_trajectory_style(swing)

    def _edit_pick_end_color(self):
        swing = self._edit_current_swing
        if not swing:
            return
        color = colorchooser.askcolor(initialcolor=swing.color_end_hex, title="終了色")
        if color[1]:
            swing.color_end_hex = color[1]
            self._edit_update_color_buttons()
            self._edit_update_display()
            self._save_trajectory_style(swing)

    def _edit_on_thickness(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.thickness = int(value)
            self.edit_thick_label.configure(text=f"{swing.thickness} px")
            self._edit_update_display()
            self._save_trajectory_style(swing)

    def _edit_on_blur(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.blur = int(value)
            self.edit_blur_label.configure(text=f"{swing.blur}")
            self._edit_update_display()
            self._save_trajectory_style(swing)

    def _edit_on_fade(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.fade_frames = int(value)
            self.edit_fade_label.configure(text=f"{swing.fade_frames} frames")
            self._edit_update_display()
            self._save_trajectory_style(swing)

    def _edit_on_alpha(self, value):
        swing = self._edit_current_swing
        if swing:
            swing.alpha = max(0.0, min(1.0, int(value) / 100.0))
            self.edit_alpha_label.configure(text=f"{int(swing.alpha * 100)}%")
            self._edit_update_display()
            self._save_trajectory_style(swing)

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
        cur = self._edit_current_swing
        if cur:
            self._save_trajectory_style(cur)
        new_swing = self._make_default_trajectory(idx)
        if cur:
            new_swing.thickness = cur.thickness
            new_swing.blur = cur.blur
            new_swing.fade_frames = cur.fade_frames
            new_swing.alpha = cur.alpha
        self._edit_swings.append(new_swing)
        self._edit_swing_idx = idx
        self._edit_update_color_buttons()
        self._edit_update_display()

    def _edit_clear_swing(self):
        swing = self._edit_current_swing
        if swing and swing.points:
            self._edit_push_undo()
            swing.points.clear()
            swing.handles.clear()
            self._edit_update_display()

    # ----- Undo / Redo (スナップショット方式) -----

    def _edit_push_undo(self):
        """現在の軌道状態を undo スタックに保存"""
        self._edit_undo_stack.append(self._edit_snapshot())
        if len(self._edit_undo_stack) > self._UNDO_MAX:
            self._edit_undo_stack.pop(0)
        self._edit_redo_stack.clear()

    def _edit_snapshot(self):
        """全スイングの points/handles のディープコピーを返す"""
        snapshot = []
        for swing in self._edit_swings:
            snapshot.append((
                list(swing.points),
                [h if h is None else tuple(h) for h in swing.handles],
            ))
        return snapshot

    def _edit_restore_snapshot(self, snapshot):
        """スナップショットから points/handles を復元"""
        for i, (pts, hds) in enumerate(snapshot):
            if i < len(self._edit_swings):
                self._edit_swings[i].points = list(pts)
                self._edit_swings[i].handles = list(hds)

    def _edit_undo(self):
        if not self._edit_undo_stack:
            return
        self._edit_redo_stack.append(self._edit_snapshot())
        snapshot = self._edit_undo_stack.pop()
        self._edit_restore_snapshot(snapshot)
        self._edit_handle_point = None
        self._edit_dragging = None
        self._edit_update_display()

    def _edit_redo(self):
        if not self._edit_redo_stack:
            return
        self._edit_undo_stack.append(self._edit_snapshot())
        snapshot = self._edit_redo_stack.pop()
        self._edit_restore_snapshot(snapshot)
        self._edit_handle_point = None
        self._edit_dragging = None
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
        self._edit_swings = [self._make_default_trajectory(0)]
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
        edit_in = self._edit_in
        edit_out = self._edit_out
        total = edit_out - edit_in + 1
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
                handles=list(s.handles) if s.handles else [],
            ))

        out_path = (self.project_dir / "exports"
                    / datetime.date.today().strftime("%m-%d") / f"swing_{clip.name}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.edit_frame_label.configure(text="書き出し中...")

        thread = threading.Thread(
            target=self._do_edit_export,
            args=(clip, swings_copy, out_path, cache, total, edit_in),
            daemon=True
        )
        thread.start()

    def _do_edit_export(self, clip, swings, out_path, cache, total, edit_in=0):
        print(f"[Edit] 動画書き出し中... ({total} frames)", flush=True)
        t0 = time.time()

        # スプライン事前構築
        spline_data = []
        for swing in swings:
            if len(swing.points) < 2:
                continue
            spline_data.append({
                "spline": TimedSpline(swing.points, SPLINE_RESOLUTION,
                                     handles=getattr(swing, 'handles', None) or None),
                "color_start": hex_to_bgr(swing.color_start_hex),
                "color_end": hex_to_bgr(swing.color_end_hex),
                "thickness": swing.thickness,
                "end_frame": getattr(swing, "end_frame", -1),
                "blur": getattr(swing, "blur", 0),
                "swing_ref": swing,
            })

        src_path = clip.exported_path if clip.exported_path else clip.source_path
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[Edit] ソースを開けません: {src_path}", flush=True)
            self.after(0, lambda: self.edit_frame_label.configure(
                text="書き出し失敗: ソースを開けません"))
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.in_frame + edit_in)

        # ====== 2パス方式: パイプI/O完全排除 ======
        # Phase 1: decode + overlay → YUV420P一時ファイル (SSD直書き)
        # Phase 2: ffmpeg がファイルからNVENCエンコード (GPU全力)
        import tempfile, os
        temp_fd, temp_raw = tempfile.mkstemp(suffix=".raw", prefix="export_")
        os.close(temp_fd)

        # リードアヘッドスレッド
        read_queue = queue.Queue(maxsize=12)

        def _reader_thread():
            try:
                for _ in range(total):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    read_queue.put(frame)
            except Exception:
                pass
            read_queue.put(None)

        read_thread = threading.Thread(target=_reader_thread, daemon=True)
        read_thread.start()

        # ライトスレッド: cvtColor + ファイル書き込みをoverlayと並列化
        write_queue = queue.Queue(maxsize=4)
        _written_count = [0]

        def _writer_thread():
            with open(temp_raw, 'wb') as f:
                while True:
                    item = write_queue.get()
                    if item is None:
                        break
                    yuv = cv2.cvtColor(item, cv2.COLOR_BGR2YUV_I420)
                    f.write(yuv.tobytes())
                    _written_count[0] += 1

        write_thread = threading.Thread(target=_writer_thread, daemon=True)
        write_thread.start()

        for i in range(total):
            frame = read_queue.get()
            if frame is None:
                break

            src_frame = i + edit_in  # 元タイムラインのフレーム番号
            for sd in spline_data:
                base_a = getattr(sd["swing_ref"], "alpha", 0.85)
                eff_alpha = _compute_fade_alpha(sd["swing_ref"], src_frame, base_alpha=base_a)
                if eff_alpha <= 0.0:
                    continue
                curve_pts = sd["spline"].get_curve_at_frame(src_frame)
                if curve_pts and len(curve_pts) >= 2:
                    full_len = len(sd["spline"]._curve)
                    ratio = len(curve_pts) / max(full_len, 1)
                    c_end = lerp_color_bgr(sd["color_start"], sd["color_end"], ratio)
                    draw_gradient_trail(frame, curve_pts, sd["color_start"],
                                        c_end, sd["thickness"],
                                        eff_alpha, blur=sd["blur"])

            write_queue.put(frame)
            if (i + 1) % 60 == 0:
                pct = int((i + 1) / total * 95)
                self.after(0, lambda p=pct: self.edit_frame_label.configure(
                    text=f"処理中... {p}%"))

        write_queue.put(None)
        write_thread.join(timeout=30)
        written = _written_count[0]
        read_thread.join(timeout=5)
        cap.release()
        t_phase1 = time.time() - t0
        print(f"[Edit] Phase1完了: {written} frames → {temp_raw} ({t_phase1:.1f}s)", flush=True)

        # Phase 2: ffmpeg でNVENCエンコード (ファイル読み → GPU全力)
        self.after(0, lambda: self.edit_frame_label.configure(text="エンコード中... 95%"))
        ffmpeg_bin = find_ffmpeg()
        if ffmpeg_bin:
            from ffmpeg_writer import _build_encoder_args
            enc_args = _build_encoder_args(
                self.settings["crf"], "fast", hw_encode=True)
            enc_cmd = [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-s", f"{clip.width}x{clip.height}",
                "-pix_fmt", "yuv420p", "-r", str(clip.fps),
                "-i", temp_raw,
            ] + enc_args + [str(out_path)]
            t_enc = time.time()
            result = subprocess.run(
                enc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            t_phase2 = time.time() - t_enc
            if result.returncode != 0:
                err = result.stderr.decode(errors="replace")[-300:]
                print(f"[Edit] エンコード失敗: {err}", flush=True)
            else:
                print(f"[Edit] Phase2完了: NVENC encode ({t_phase2:.1f}s)", flush=True)
        else:
            # ffmpeg なし: cv2 フォールバック (Phase1のrawから再読み込み)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w2 = cv2.VideoWriter(str(out_path), fourcc, clip.fps,
                                  (clip.width, clip.height))
            frame_bytes = clip.width * clip.height * 3
            with open(temp_raw, 'rb') as f:
                while True:
                    raw = f.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        break
                    fr = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (clip.height, clip.width, 3))
                    w2.write(fr)
            w2.release()

        # 一時ファイル削除
        try:
            os.remove(temp_raw)
        except OSError:
            pass

        elapsed = time.time() - t0
        print(f"[Edit] 出力: {out_path} ({written}/{total} frames, {elapsed:.1f}s)", flush=True)
        # 送出リストに自動追加 (GUIスレッドで実行)
        self.after(0, lambda p=out_path: self._add_export_to_playout(str(p)))

    def _add_export_to_playout(self, export_path):
        """書き出した動画を送出リストに自動追加 (クリップには追加しない)
        同じファイルパスのエントリが既にあれば上書き (重複防止)"""
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
            # 同じファイルパスの既存エントリを除去 (再書き出し時の重複防止)
            resolved = str(p.resolve())
            self.playout.playlist = [
                item for item in self.playout.playlist
                if str(Path(item.clip.source_path).resolve()) != resolved
            ]
            self.playout.add_item(clip, [])
            self.playout.save_playlist(self._playout_json)
            self._playout_dirty = True
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

        self.po_rec_btn = ctk.CTkButton(
            right, text="⏺ REC", width=250, height=36,
            font=("", 14, "bold"),
            fg_color="#8B0000", hover_color="#B22222",
            command=self._toggle_rec
        )
        self.po_rec_btn.pack(padx=10, pady=(8, 2))

        pl_header = ctk.CTkFrame(right, fg_color="transparent")
        pl_header.pack(fill="x", padx=5, pady=(10, 0))
        ctk.CTkLabel(pl_header, text="送出リスト", font=("", 16, "bold")).pack(side="left", padx=5)
        ctk.CTkButton(
            pl_header, text="全消し", width=60, height=28,
            fg_color="#8B0000", hover_color="#A52A2A",
            font=("", 12, "bold"),
            command=self._playout_clear,
        ).pack(side="right", padx=5)

        self.playout_scroll = ctk.CTkScrollableFrame(right)
        self.playout_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.playout_scroll.grid_columnconfigure(0, weight=1)
        self._playout_widgets = []
        self._playout_row_map = {}  # playlist index → row widget

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
        self._playout_dirty = False
        for w in self._playout_widgets:
            w.destroy()
        self._playout_widgets.clear()

        # 日付グループ表示用: ソースパスから日付ディレクトリを取得
        def _po_date_key(item):
            parent = Path(item.clip.source_path).parent.name
            if (len(parent) == 5 and parent[2] == '-') or (len(parent) == 10 and parent[4] == '-' and parent[7] == '-'):
                return parent
            return ""

        current_group = None
        collapsed = False
        grid_row = 0
        self._playout_row_map = {}  # playlist index → row widget
        for i, item in enumerate(self.playout.playlist):
            group = _po_date_key(item)
            if group != current_group:
                current_group = group
                collapsed = group in self._collapsed_playout_groups
                arrow = "▶" if collapsed else "▼"
                label = group if group else "その他"
                header = ctk.CTkFrame(self.playout_scroll, height=22,
                                      fg_color="#1a1a3a", cursor="hand2")
                header.grid(row=grid_row, column=0, sticky="ew", pady=(4, 1))
                header.grid_columnconfigure(0, weight=1)
                hdr_btn = ctk.CTkButton(
                    header, text=f"{arrow} {label}",
                    font=("", 11, "bold"), text_color="#8888CC",
                    fg_color="transparent", hover_color="#2a2a4a",
                    anchor="w",
                    command=lambda g=group: self._toggle_playout_group(g))
                hdr_btn.grid(row=0, column=0, sticky="ew", padx=4)
                dir_path = str(Path(item.clip.source_path).parent)
                hdr_btn.bind("<Button-3>",
                             lambda e, d=dir_path: self._show_folder_menu(e, d))
                self._playout_widgets.append(header)
                grid_row += 1

            if collapsed:
                continue

            row = ctk.CTkFrame(self.playout_scroll, height=35)
            row.grid(row=grid_row, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            # 選択中ハイライト
            is_selected = (self._playout_selected_idx == i)
            if is_selected:
                row.configure(fg_color="#1a3a1a", border_color="#00AA00", border_width=1)

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
            self._playout_row_map[i] = row
            grid_row += 1

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
        # GUI更新は on_frame_update → _po_flush_frame で行われる

    def _playout_seek_delta(self, delta):
        """送出タブでフレーム送り/戻り (停止中 or 一時停止中)"""
        if self.playout._playing and not self.playout._paused:
            return
        total = self.playout.get_cued_total_frames()
        if total <= 0:
            return
        current = int(self.po_seek_slider.get())
        new_pos = max(0, min(current + delta, total - 1))
        self.playout.seek_to(new_pos)
        # GUI更新は on_frame_update → _po_flush_frame で行われる

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
        for pi, w in self._playout_row_map.items():
            if pi == idx:
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
        self._playout_dirty = True
        self._refresh_playout_list()

    def _playout_clear(self):
        """送出リストを全消し (実ファイルも削除)"""
        if not self.playout.playlist:
            return
        # 実ファイル削除
        for item in self.playout.playlist:
            src = Path(item.clip.source_path)
            if src.exists():
                try:
                    src.unlink()
                    print(f"[Playout] ファイル削除: {src.name}")
                except Exception as e:
                    print(f"[Playout] ファイル削除エラー: {src.name}: {e}")
        self.playout.playlist.clear()
        self._playout_selected_idx = None
        self.playout.save_playlist(self._playout_json)
        self._playout_dirty = True
        self._refresh_playout_list()

    def _playout_open_in_edit(self, idx):
        """送出リストのアイテムを編集タブで開く"""
        if idx < 0 or idx >= len(self.playout.playlist):
            return
        item = self.playout.playlist[idx]
        clip = item.clip

        print(f"[Edit] 送出クリップ読み込み: {clip.name}")
        if self._edit_slider_after is not None:
            self.after_cancel(self._edit_slider_after)
            self._edit_slider_after = None
        self._edit_clip_id = None
        self._edit_direct_clip = clip
        self._edit_cache = FrameCache(clip.source_path, clip.in_frame, clip.get_out_frame())
        self._edit_total = len(self._edit_cache)
        self._edit_frame_no = 0

        self._edit_swings = [self._make_default_trajectory(0)]
        self._edit_swing_idx = 0

        self.edit_slider.configure(to=max(self._edit_total - 1, 1))
        self._edit_reset_in_out()
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
        outer = self.tab_settings
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        tab = ctk.CTkScrollableFrame(outer)
        tab.pack(fill="both", expand=True)

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

        # CRF (録画品質)
        crf_row = ctk.CTkFrame(sec3, fg_color="transparent")
        crf_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(crf_row, text="録画品質 (CRF):").pack(side="left", padx=(0, 5))
        self._crf_value_label = ctk.CTkLabel(crf_row, text=str(self.settings["crf"]),
                                              width=30)
        self._crf_value_label.pack(side="right", padx=(5, 0))
        ctk.CTkLabel(crf_row, text="低品質", font=("", 10),
                      text_color="#888").pack(side="right", padx=(5, 0))
        self.crf_slider = ctk.CTkSlider(
            crf_row, from_=0, to=28, number_of_steps=28, width=200,
            command=self._on_crf_slider)
        self.crf_slider.set(self.settings["crf"])
        self.crf_slider.pack(side="right", padx=(5, 0))
        ctk.CTkLabel(crf_row, text="高品質", font=("", 10),
                      text_color="#888").pack(side="right", padx=(5, 0))

        # グローウィングバッファ時間
        buf_row = ctk.CTkFrame(sec3, fg_color="transparent")
        buf_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(buf_row, text="グローウィング最大時間:").pack(side="left", padx=(0, 5))
        self.growing_buf_combo = ctk.CTkComboBox(
            buf_row,
            values=["30秒", "1分", "2分", "3分", "5分", "10分"],
            width=100,
            command=self._on_growing_buf_changed,
        )
        # 現在の設定値から表示を復元
        cur_sec = self.settings["growing_buffer_sec"]
        buf_labels = {30: "30秒", 60: "1分", 120: "2分", 180: "3分", 300: "5分", 600: "10分"}
        self.growing_buf_combo.set(buf_labels.get(cur_sec, f"{cur_sec}秒"))
        self.growing_buf_combo.pack(side="left", padx=(0, 10))
        self._growing_mem_label = ctk.CTkLabel(buf_row, text="",
                                                font=("", 10), text_color="#888")
        self._growing_mem_label.pack(side="left")
        self._update_growing_mem_label(cur_sec)

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

        # キーボードショートカット設定
        self._build_keyboard_shortcut_settings(tab)

        # ShuttlePRO v2 ボタン設定
        self._build_shuttle_settings(tab)

        # 保存ボタン
        ctk.CTkButton(tab, text="設定を保存", width=200, height=40,
                       font=("", 14, "bold"),
                       fg_color="#006400", hover_color="#228B22",
                       command=self._save_settings).pack(pady=20)

    def _build_keyboard_shortcut_settings(self, tab):
        """キーボードショートカット設定セクション"""
        sec = ctk.CTkFrame(tab)
        sec.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(sec, text="キーボードショートカット",
                      font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        ctk.CTkLabel(sec, text="ボタンをクリックしてキーを押すと変更できます",
                      font=("", 10), text_color="#888").pack(anchor="w", padx=10, pady=(0, 5))

        shortcuts = self.settings.data.get("keyboard_shortcuts",
                                            dict(DEFAULT_KEYBOARD_SHORTCUTS))
        self._kb_shortcut_btns = {}  # action_id -> CTkButton
        self._kb_capture_active = None  # 現在キャプチャ中の action_id

        # タブコンテキストでグループ分け
        groups = {}
        for action_id, label, default_key, context in KEYBOARD_ACTIONS:
            groups.setdefault(context, []).append((action_id, label, default_key))

        content = ctk.CTkFrame(sec, fg_color="transparent")
        content.pack(fill="x", padx=10, pady=5)

        col_idx = 0
        for context, actions in groups.items():
            grp = ctk.CTkFrame(content)
            grp.pack(side="left", fill="y", padx=5, pady=2, anchor="n")
            ctk.CTkLabel(grp, text=context, font=("", 11, "bold"),
                          text_color="#aaa").pack(anchor="w", padx=5, pady=(5, 2))

            for action_id, label, default_key in actions:
                row = ctk.CTkFrame(grp, fg_color="transparent")
                row.pack(fill="x", padx=5, pady=1)
                ctk.CTkLabel(row, text=label, width=130,
                              anchor="w", font=("", 11)).pack(side="left")
                current_key = shortcuts.get(action_id, default_key)
                display = self._keysym_display(current_key)
                btn = ctk.CTkButton(
                    row, text=display, width=70, height=24,
                    font=("", 11), fg_color="#333", hover_color="#555",
                    command=lambda aid=action_id: self._kb_start_capture(aid))
                btn.pack(side="left", padx=3)
                self._kb_shortcut_btns[action_id] = btn

            col_idx += 1

    @staticmethod
    def _keysym_display(keysym):
        """keysymを表示用文字列に変換"""
        if keysym in KEYSYM_DISPLAY:
            return KEYSYM_DISPLAY[keysym]
        if keysym.startswith("F") and keysym[1:].isdigit():
            return keysym  # F1-F12
        if len(keysym) == 1:
            return keysym.upper()
        return keysym

    def _kb_start_capture(self, action_id):
        """キーキャプチャモードを開始"""
        # 前回のキャプチャをキャンセル
        if self._kb_capture_active and self._kb_capture_active in self._kb_shortcut_btns:
            prev_btn = self._kb_shortcut_btns[self._kb_capture_active]
            shortcuts = self.settings.data.get("keyboard_shortcuts",
                                                dict(DEFAULT_KEYBOARD_SHORTCUTS))
            prev_key = shortcuts.get(self._kb_capture_active, "")
            prev_btn.configure(text=self._keysym_display(prev_key), fg_color="#333")

        self._kb_capture_active = action_id
        btn = self._kb_shortcut_btns[action_id]
        btn.configure(text="...", fg_color="#8B4513")
        # bind_all で次のキー入力をキャプチャ
        self._kb_capture_bind_id = self.bind_all("<Key>", self._kb_on_capture, add=False)

    def _kb_on_capture(self, event):
        """キーキャプチャ: 押されたキーをアクションに割り当て"""
        import tkinter as tk
        if not self._kb_capture_active:
            return

        keysym = event.keysym
        action_id = self._kb_capture_active
        self._kb_capture_active = None

        # キャプチャ用バインドを解除して通常キーバインドに戻す
        self.unbind_all("<Key>")
        self._bind_global_keys()

        # ショートカットを更新
        shortcuts = self.settings.data.get("keyboard_shortcuts",
                                            dict(DEFAULT_KEYBOARD_SHORTCUTS))
        # 同じタブ内の重複チェック
        my_context = None
        for aid, _, _, ctx in KEYBOARD_ACTIONS:
            if aid == action_id:
                my_context = ctx
                break
        for aid, _, _, ctx in KEYBOARD_ACTIONS:
            if aid != action_id and shortcuts.get(aid) == keysym and ctx == my_context:
                # 重複: 古い方をクリア
                shortcuts[aid] = ""
                if aid in self._kb_shortcut_btns:
                    self._kb_shortcut_btns[aid].configure(text="-")

        shortcuts[action_id] = keysym
        self.settings["keyboard_shortcuts"] = shortcuts
        self.settings.save()

        # ボタン表示を更新
        btn = self._kb_shortcut_btns[action_id]
        btn.configure(text=self._keysym_display(keysym), fg_color="#333")

        # キーマッピングを再構築
        self._rebuild_key_map()

    def _rebuild_key_map(self):
        """settings からキーマッピングのルックアップテーブルを再構築"""
        shortcuts = self.settings.data.get("keyboard_shortcuts",
                                            dict(DEFAULT_KEYBOARD_SHORTCUTS))
        # tab_context -> {keysym_lower: action_id}
        self._key_map = {}
        for action_id, _, default_key, context in KEYBOARD_ACTIONS:
            keysym = shortcuts.get(action_id, default_key)
            if not keysym:
                continue
            self._key_map.setdefault(context, {})[keysym.lower()] = action_id

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

    def _on_growing_buf_changed(self, choice):
        buf_map = {"30秒": 30, "1分": 60, "2分": 120, "3分": 180, "5分": 300, "10分": 600}
        self._update_growing_mem_label(buf_map.get(choice, 60))

    def _update_growing_mem_label(self, sec):
        """グローウィングバッファの推定メモリ使用量を表示"""
        try:
            fps = float(self.fps_combo.get())
        except (ValueError, AttributeError):
            fps = self.settings["fps"]
        try:
            res = self.resolution_combo.get()
            w, h = (int(x) for x in res.split("x"))
        except (ValueError, AttributeError):
            w, h = self.settings["width"], self.settings["height"]
        frames = int(sec * fps)
        # プレビュー: 480×270×3 bytes/frame, フルレスJPEG: 解像度依存 (Q97で約6-8%)
        preview_bytes = frames * 480 * 270 * 3
        jpeg_bytes = frames * int(w * h * 3 * 0.07)
        total_mb = (preview_bytes + jpeg_bytes) / (1024 * 1024)
        if total_mb >= 1024:
            self._growing_mem_label.configure(text=f"推定メモリ: 約 {total_mb / 1024:.1f} GB")
        else:
            self._growing_mem_label.configure(text=f"推定メモリ: 約 {total_mb:.0f} MB")

    def _on_crf_slider(self, value):
        self._crf_value_label.configure(text=str(int(value)))

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

        self.settings["crf"] = int(self.crf_slider.get())

        # グローウィングバッファ時間
        buf_text = self.growing_buf_combo.get()
        buf_map = {"30秒": 30, "1分": 60, "2分": 120, "3分": 180, "5分": 300, "10分": 600}
        self.settings["growing_buffer_sec"] = buf_map.get(buf_text, 60)

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

        # キーボードショートカット (キャプチャ時にも保存されるが、念のため)
        if hasattr(self, "_kb_shortcut_btns"):
            shortcuts = self.settings.data.get("keyboard_shortcuts",
                                                dict(DEFAULT_KEYBOARD_SHORTCUTS))
            self.settings["keyboard_shortcuts"] = shortcuts

        # プロジェクトフォルダ変更の反映
        new_project = Path(self.settings["project_dir"])
        if new_project != self.project_dir:
            old_rec_default = str(self.project_dir / "recordings")
            new_project.mkdir(parents=True, exist_ok=True)
            self.project_dir = new_project
            self.settings.path = new_project / "settings.json"
            self.clip_manager = ClipManager(str(new_project))
            self._exports_dir = new_project / "exports"
            self._exports_dir.mkdir(parents=True, exist_ok=True)
            self._playout_json = str(new_project / "playout.json")
            self.playout.load_playlist(self._playout_json)
            self.playout.scan_directory(str(self._exports_dir))
            self.playout.save_playlist(self._playout_json)
            # record_dir が旧プロジェクトのデフォルトだった場合、新プロジェクトに追従
            if os.path.normpath(self.settings["record_dir"]) == os.path.normpath(old_rec_default):
                new_rec = str(new_project / "recordings")
                self.settings["record_dir"] = new_rec
                self.record_dir_entry.delete(0, "end")
                self.record_dir_entry.insert(0, new_rec)
            print(f"[Settings] プロジェクトフォルダ変更: {new_project}")

            # デフォルト場所にもproject_dirを保存 (次回起動時に新フォルダを見つけるため)
            default_settings = Path(DEFAULT_PROJECT_DIR) / "settings.json"
            if default_settings != self.settings.path:
                try:
                    default_settings.parent.mkdir(parents=True, exist_ok=True)
                    import json
                    default_data = {}
                    if default_settings.exists():
                        with open(default_settings, "r", encoding="utf-8") as f:
                            default_data = json.load(f)
                    default_data["project_dir"] = str(new_project)
                    with open(default_settings, "w", encoding="utf-8") as f:
                        json.dump(default_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[Settings] デフォルト設定の更新エラー: {e}")

        self.settings.save()
        self._rebuild_key_map()
        self.recorder = Recorder(
            self.settings["record_dir"],
            self.settings["width"],
            self.settings["height"],
            self.settings["fps"],
            crf=self.settings["crf"],
            growing_buffer_sec=self.settings["growing_buffer_sec"],
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
        self._jog_pending_delta = 0      # ジョグ蓄積デルタ
        self._jog_timer = None           # ジョグスロットルタイマー

        def on_jog(delta):
            self._jog_pending_delta += delta
            if self._jog_timer is None:
                self._jog_timer = self.after(33, self._flush_jog)  # ~30fps上限

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

    def _flush_jog(self):
        """蓄積されたジョグデルタをまとめて1回で処理"""
        self._jog_timer = None
        delta = self._jog_pending_delta
        self._jog_pending_delta = 0
        if delta != 0:
            self._shuttle_jog(delta)

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

        # キーマッピングを構築
        if not hasattr(self, "_key_map") or not self._key_map:
            self._rebuild_key_map()

        # タブ名→コンテキスト名マッピング
        tab_contexts = {
            "クリップ": "クリップ",
            "編集": "編集",
            "送出": "送出",
            "収録": "収録",
        }

        # アクション→実行関数マッピング
        action_handlers = {
            "frame_fwd":       lambda: self._dispatch_frame_jump(self._frame_step),
            "frame_back":      lambda: self._dispatch_frame_jump(-self._frame_step),
            "frame_fwd_fast":  lambda: self._dispatch_frame_jump(self._frame_step * 5),
            "frame_back_fast": lambda: self._dispatch_frame_jump(-self._frame_step * 5),
            "step_1":          lambda: self._on_frame_step_change("1"),
            "step_2":          lambda: self._on_frame_step_change("2"),
            "step_5":          lambda: self._on_frame_step_change("5"),
            "step_10":         lambda: self._on_frame_step_change("10"),
            "set_in":          lambda: self._set_in_current(),
            "set_out":         lambda: self._set_out_current(),
            "edit_play":       lambda: self._edit_toggle_play(),
            "edit_set_in":     lambda: self._edit_set_in(),
            "edit_set_out":    lambda: self._edit_set_out(),
            "zoom_reset":      lambda: self._edit_zoom_reset(),
            "po_play_pause":   lambda: self._playout_toggle_play(),
            "po_play":         lambda: self._playout_play_with_btn(),
            "po_cue_top":      lambda: self._playout_cue_top(),
            "po_next":         lambda: self._playout_next(),
            "po_prev":         lambda: self._playout_prev(),
            "po_speed_1x":     lambda: self._set_playout_speed("1x"),
            "po_speed_1_2":    lambda: self._set_playout_speed("1/2"),
            "po_speed_1_4":    lambda: self._set_playout_speed("1/4"),
            "po_speed_1_8":    lambda: self._set_playout_speed("1/8"),
            "toggle_rec":      lambda: self._toggle_rec(),
        }

        def on_key(event):
            w = event.widget
            if isinstance(w, (tk.Entry, ctk.CTkEntry)):
                return

            key = event.keysym.lower()
            current_tab = self.tabview.get()
            context = tab_contexts.get(current_tab)
            if not context:
                return

            # Ctrl+Z / Ctrl+Shift+Z (編集タブ Undo/Redo)
            ctrl = event.state & 0x4
            shift = event.state & 0x1
            if ctrl and key == "z" and context == "編集":
                if shift:
                    self._edit_redo()
                else:
                    self._edit_undo()
                return

            # タブ固有のショートカットを先にチェック
            tab_map = self._key_map.get(context, {})
            action = tab_map.get(key)

            # 共通ショートカット
            if not action:
                common_map = self._key_map.get("共通", {})
                action = common_map.get(key)

            if action and action in action_handlers:
                action_handlers[action]()

        self.bind_all("<Key>", on_key)

    def _dispatch_frame_jump(self, delta):
        """現在のタブに応じてフレームジャンプを実行"""
        current_tab = self.tabview.get()
        if current_tab == "クリップ":
            self._clip_jump(delta)
        elif current_tab == "編集":
            self._edit_jump(delta)
        elif current_tab == "送出":
            self._playout_seek_delta(delta)

    def _edit_zoom_reset(self):
        """編集タブのズームをリセット"""
        self._edit_zoom = 1.0
        self._edit_pan_vx = 0.0
        self._edit_pan_vy = 0.0
        self._edit_update_display()

    def _playout_play_with_btn(self):
        """送出再生 + ボタン表示更新"""
        self._playout_play()
        self.po_play_btn.configure(text="⏸ PAUSE",
            fg_color="#B8860B", hover_color="#DAA520")

    def _set_playout_speed(self, speed):
        """送出速度を設定 + UI同期"""
        self.po_speed_seg.set(speed)
        self._on_speed_change(speed)

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
        self._release_clip_preview_cap()
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
