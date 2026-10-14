"""ai.edit_hook — 編集タブに「ヘッド軌道を自動生成」「縦型動画 (SNS) 書き出し」を足す。

app.py からは `attach_edit_tools(self, edit_scroll)` を 1 回呼ぶだけ。
自動生成した軌道は ClipManager.save_trajectory() で手描きと同じ
trajectories/{clip_id}_trajectory.json に保存するので、編集タブの既存機能
(点の移動・削除・ベジェハンドル・Undo・書き出し) がそのまま使える。

app の内部状態への依存 (ここだけに閉じ込める):
    _edit_clip_id / _edit_swings / _edit_swing_idx / _edit_in / _edit_out
    _edit_push_undo() / _edit_spline_cache / _edit_update_display() / _edit_update_color_buttons()
    _clips_tab_dirty / _edit_tab_dirty
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

import customtkinter as ctk

from clip_manager import TrajectoryData
from ai import ai_available
from ai.config import get_ai_settings, realtime_factor
from ai.store import analysis_dir, load_club_track, landmarks_from_track, save_club_track


class EditTools:
    def __init__(self, app, parent):
        self.app = app
        self._busy = False

        sec = ctk.CTkFrame(parent)
        sec.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(sec, text="AI (試作)", font=("", 12, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
        self.trace_btn = ctk.CTkButton(
            sec, text="ヘッド軌道を自動生成", height=30,
            fg_color="#2E4057", hover_color="#3D5A80",
            command=self.run_autotrace)
        self.trace_btn.pack(fill="x", padx=8, pady=(4, 2))
        self.sns_btn = ctk.CTkButton(
            sec, text="縦型動画 (SNS) 書き出し", height=30,
            fg_color="#4A3B6B", hover_color="#5E4D86",
            command=self.run_sns_export)
        self.sns_btn.pack(fill="x", padx=8, pady=2)
        self.status = ctk.CTkLabel(sec, text="", font=("", 10), text_color="#AAAAAA",
                                   wraplength=200, justify="left")
        self.status.pack(anchor="w", padx=8, pady=(2, 6))
        if not ai_available():
            self.trace_btn.configure(state="disabled")
            self.sns_btn.configure(state="disabled")
            self.status.configure(text="mediapipe が未インストール")

    # ---------------------------------------------------------------- 共通
    def _set_status(self, text, color="#AAAAAA"):
        self.app.after(0, lambda: self.status.configure(text=text, text_color=color))

    def _current_clip(self):
        cid = getattr(self.app, "_edit_clip_id", None)
        if not cid:
            self._set_status("編集タブでクリップを選択してください", "#E0B040")
            return None
        clip = self.app.clip_manager.get_clip(cid)
        if clip is None or not Path(clip.source_path).exists():
            self._set_status("クリップのファイルが見つかりません", "#E0704B")
            return None
        return clip

    def _start(self, worker, label):
        if self._busy:
            self._set_status("処理中です", "#E0B040")
            return
        self._busy = True
        self.trace_btn.configure(state="disabled")
        self.sns_btn.configure(state="disabled")
        self._set_status(f"{label}...")

        def run():
            try:
                worker()
            except Exception as e:
                traceback.print_exc()
                self._set_status(f"{label}に失敗: {e}", "#E0704B")
            finally:
                self._busy = False
                self.app.after(0, lambda: (self.trace_btn.configure(state="normal"),
                                           self.sns_btn.configure(state="normal")))

        threading.Thread(target=run, daemon=True, name=f"ai-{label}").start()

    # ---------------------------------------------------------------- 自動トレース
    def run_autotrace(self):
        clip = self._current_clip()
        if clip is None:
            return
        app = self.app
        in_f, out_f = clip.in_frame, clip.get_out_frame()
        style = app.settings.data.get("trajectory_style", {})

        def worker():
            from ai.club_tracker import track_club
            res = track_club(clip.source_path, in_f, out_f,
                             progress=lambda r, m: self._set_status(f"{m} {r * 100:.0f}%"))
            save_club_track(analysis_dir(app.project_dir, clip.id), res, in_f, clip.source_path)
            ctrl = [(int(x), int(y), f - in_f) for x, y, f in res.confident_points()]
            weak = [(a - in_f, b - in_f) for a, b in res.weak_ranges()]
            app.after(0, lambda: self._apply_trajectory(clip, ctrl, weak, style, res))

        self._start(worker, "軌道生成")

    def _apply_trajectory(self, clip, ctrl, weak, style, res):
        """GUI スレッド: 生成した制御点を軌道として保存し、編集タブに反映する。"""
        app = self.app
        editing = getattr(app, "_edit_clip_id", None) == clip.id
        if editing:
            swings = app._edit_swings
            edit_in, edit_out = app._edit_in, app._edit_out
            app._edit_push_undo()
        else:
            swings, edit_in, edit_out = app.clip_manager.load_trajectory(clip.id)
        new_swing = TrajectoryData(
            points=ctrl,
            color_start_hex=style.get("color_start_hex", "#FFFF00"),
            color_end_hex=style.get("color_end_hex", "#FF0000"),
            thickness=style.get("thickness", 3), blur=style.get("blur", 0),
            fade_frames=style.get("fade_frames", 0), alpha=style.get("alpha", 0.85),
            handles=[None] * len(ctrl))
        # 空のスイングがあればそこに入れ、無ければ新しいスイングとして追加 (手描きは消さない)
        idx = next((i for i, s in enumerate(swings) if len(s.points) < 2), None)
        if idx is None:
            swings.append(new_swing)
            idx = len(swings) - 1
        else:
            swings[idx] = new_swing
        app.clip_manager.save_trajectory(clip.id, swings, edit_in=edit_in, edit_out=edit_out)
        if editing:
            app._edit_swings = swings
            app._edit_spline_cache.clear()
            app._edit_swing_idx = idx
            app._edit_update_color_buttons()
            app._edit_update_display()
        app._clips_tab_dirty = app._edit_tab_dirty = True
        msg = (f"完了 ({res.seconds:.0f}秒): 制御点 {len(ctrl)} 点 → Swing {idx + 1}"
               f" (観測 {res.frames_detected}/{res.frames_total}f)")
        if weak:
            msg += " / 要手直し F" + " ".join(f"{a}-{b}" for a, b in weak[:3])
        self._set_status(msg, "#E0B040" if weak else "#88CC88")

    # ---------------------------------------------------------------- 縦型書き出し
    def run_sns_export(self):
        clip = self._current_clip()
        if clip is None:
            return
        app = self.app
        # 編集中の軌道を先に保存して、手直し後の点を使う
        if getattr(app, "_edit_clip_id", None) == clip.id:
            app._edit_autosave_trajectory()
        swings, edit_in, edit_out = app.clip_manager.load_trajectory(clip.id)
        swings = [s for s in swings if len(s.points) >= 2]
        if not swings:
            self._set_status("軌道がありません。先に自動生成か手描きをしてください", "#E0B040")
            return
        cfg = get_ai_settings(app.settings)
        out_dir = app.project_dir / "exports" / "sns"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip.name}_short.mp4"
        track = load_club_track(analysis_dir(app.project_dir, clip.id))
        landmarks = landmarks_from_track(track)
        dense = [tuple(p) for p in track.get("points", [])] if track else None
        factor = realtime_factor(clip.fps, cfg["source_kind"])

        def worker():
            from ai.sns_render import render_sns
            rep = render_sns(
                clip.source_path, out_path, in_frame=clip.in_frame, out_frame=clip.get_out_frame(),
                swings=swings, landmarks=landmarks, realtime_factor=factor, options=cfg["sns"],
                dense_points=dense, camera_angle=cfg.get("camera_angle", "auto"),
                progress=lambda r, m: self._set_status(f"{m} {r * 100:.0f}%"))
            self._set_status(f"完了: {out_path.name} ({rep['frames']}f, {rep['seconds']:.0f}秒)"
                             f"\n{out_dir}", "#88CC88")

        self._start(worker, "縦型書き出し")


def attach_edit_tools(app, parent) -> EditTools:
    return EditTools(app, parent)
