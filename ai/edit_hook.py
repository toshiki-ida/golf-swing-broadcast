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
from ai.dataset import collect_clip_labels, dataset_dir
from ai.store import (analysis_dir, load_club_track, landmarks_from_track, save_club_track,
                      load_clip_ai_settings, save_clip_ai_settings, save_auto_snapshot, record_corrections)


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
        row = ctk.CTkFrame(sec, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(row, text="撮影方向", font=("", 11)).pack(side="left")
        self.camera_seg = ctk.CTkSegmentedButton(row, values=["auto", "dtl", "face_on"], width=150,
                                                 font=("", 11), command=self._on_camera_changed)
        self.camera_seg.set("auto")
        self.camera_seg.pack(side="left", padx=6)
        self.undrawn_label = ctk.CTkLabel(sec, text="", font=("", 10), text_color="#E0B040",
                                          wraplength=200, justify="left")
        self.undrawn_label.pack(anchor="w", padx=8, pady=(4, 0))
        row2 = ctk.CTkFrame(sec, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(2, 0))
        self.jump_btn = ctk.CTkButton(row2, text="次の未描画区間へ", width=120, height=24, font=("", 11),
                                      command=self.jump_to_undrawn)
        self.jump_btn.pack(side="left")
        self.merge_btn = ctk.CTkButton(row2, text="区間を結合", width=80, height=24, font=("", 11),
                                       command=self.merge_segments)
        self.merge_btn.pack(side="left", padx=4)
        self.status = ctk.CTkLabel(sec, text="", font=("", 10), text_color="#AAAAAA",
                                   wraplength=200, justify="left")
        self.status.pack(anchor="w", padx=8, pady=(2, 6))
        self._undrawn: dict = {}      # clip_id -> [(a, b), ...] 未描画区間 (in_frame 基準の相対フレーム)
        self._undrawn_idx = 0
        self._wrap_save_trajectory()
        self._wrap_edit_export()
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

    # ---------------------------------------------------------------- 未描画区間 / 手直しの記録
    def _wrap_save_trajectory(self):
        """ClipManager.save_trajectory を包み、保存のたびに corrections.json を更新する (app.py は変更しない)"""
        cm = self.app.clip_manager
        if getattr(cm, "_ai_wrapped", False):
            return
        orig = cm.save_trajectory

        def wrapped(clip_id, swings, edit_in=0, edit_out=-1, _orig=orig):
            _orig(clip_id, swings, edit_in=edit_in, edit_out=edit_out)
            try:
                record_corrections(analysis_dir(self.app.project_dir, clip_id), swings)
            except Exception as e:
                print(f"[AI] corrections の記録に失敗: {e}")

        cm.save_trajectory = wrapped
        cm._ai_wrapped = True

    def _collect_dataset(self, clip, swings, label=""):
        """書き出し時: そのクリップの軌道を承認済みラベルとして datasets/clubhead に保存 (holdout は除外、再書き出しは上書き)"""
        cfg = get_ai_settings(self.app.settings)
        if not cfg.get("collect_labels", True):
            return None
        try:
            import json as _j
            adir = analysis_dir(self.app.project_dir, clip.id)
            track = load_club_track(adir)
            corr_p = adir / "corrections.json"
            corr = _j.loads(corr_p.read_text(encoding="utf-8")) if corr_p.exists() else None
            lm = landmarks_from_track(track)
            body_h = (track or {}).get("body_h")
            real_fps = (track or {}).get("real_fps") or clip.fps * realtime_factor(clip.fps, cfg["source_kind"])
            res = collect_clip_labels(clip.id, clip.source_path, clip.in_frame, swings, corrections=corr,
                                      landmarks=lm, body_h=body_h, real_fps=real_fps, labeler="operator",
                                      note=f"{label}書き出し時に承認", out_root=dataset_dir(self.app.project_dir))
            if res is None:
                print(f"[AI] ラベル収集: {clip.name} は holdout のため除外 (または点が 2 点未満)")
            else:
                print(f"[AI] ラベル収集: {clip.name} → strong {res['n_strong']} / weak {res['n_weak']} {res['sources']} → {res['dir']}")
            return res
        except Exception as e:
            print(f"[AI] ラベル収集に失敗: {e}")
            return None

    def _wrap_edit_export(self):
        """app._do_edit_export (送出用の軌道焼き込み書き出し) を包み、書き出し完了後にラベルを収集する"""
        app = self.app
        if getattr(app, "_ai_export_wrapped", False) or not hasattr(app, "_do_edit_export"):
            return
        orig = app._do_edit_export

        def wrapped(clip, swings, out_path, cache, total, edit_in=0, _orig=orig):
            _orig(clip, swings, out_path, cache, total, edit_in)
            self._collect_dataset(clip, [s for s in swings if len(s.points) >= 2], "送出用")

        app._do_edit_export = wrapped
        app._ai_export_wrapped = True

    def _show_undrawn(self, clip_id):
        spans = self._undrawn.get(clip_id) or []
        if not spans:
            self.undrawn_label.configure(text="")
            return
        txt = "未描画区間 (手描きで補える): " + " ".join(f"F{a}-{b}" for a, b in spans[:6])
        if len(spans) > 6:
            txt += f" 他 {len(spans) - 6}"
        self.undrawn_label.configure(text=txt)

    def jump_to_undrawn(self):
        """未描画区間の先頭フレームへ順番に移動する (編集タブのスライダーと表示を更新)"""
        app = self.app
        cid = getattr(app, "_edit_clip_id", None)
        spans = self._undrawn.get(cid) or []
        if not spans:
            self._set_status("未描画区間はありません")
            return
        a, b = spans[self._undrawn_idx % len(spans)]
        self._undrawn_idx += 1
        app._edit_frame_no = max(0, min(int(a), app._edit_total - 1))
        app._edit_update_display()
        self._set_status(f"未描画区間 F{a}-{b} の先頭へ移動 (左クリックで点を足す)")

    def merge_segments(self):
        """自動生成の区間と手描きの点を 1 本の Swing に結合して連続表示にする (Undo 可)"""
        app = self.app
        cid = getattr(app, "_edit_clip_id", None)
        if not cid or not app._edit_swings:
            return
        app._edit_push_undo()
        pts = sorted({p for sw in app._edit_swings for p in sw.points}, key=lambda q: q[2])
        base = app._edit_swings[0]
        merged = TrajectoryData(points=pts, color_start_hex=base.color_start_hex, color_end_hex=base.color_end_hex,
                                thickness=base.thickness, blur=getattr(base, "blur", 0),
                                fade_frames=getattr(base, "fade_frames", 0), alpha=getattr(base, "alpha", 0.85),
                                handles=[None] * len(pts))
        app._edit_swings = [merged]
        app._edit_swing_idx = 0
        app._edit_spline_cache.clear()
        app._edit_update_color_buttons()
        app._edit_update_display()
        app._edit_autosave_trajectory()
        self._set_status(f"{len(pts)} 点を 1 本の軌道に結合しました (未描画区間も補間で繋がります)")

    def _on_camera_changed(self, value):
        """撮影方向をクリップごとに保存 (auto / dtl / face_on)"""
        cid = getattr(self.app, "_edit_clip_id", None)
        if cid:
            save_clip_ai_settings(analysis_dir(self.app.project_dir, cid), camera_angle=value)
            self._set_status(f"撮影方向: {value} (このクリップに保存)")

    def camera_angle_for(self, clip_id) -> str:
        """クリップごとの設定 → 無ければ settings.json の ai.camera_angle"""
        saved = load_clip_ai_settings(analysis_dir(self.app.project_dir, clip_id)).get("camera_angle")
        if saved:
            self.camera_seg.set(saved)
            return saved
        return get_ai_settings(self.app.settings).get("camera_angle", "auto")

    # ---------------------------------------------------------------- 自動トレース
    def run_autotrace(self):
        clip = self._current_clip()
        if clip is None:
            return
        app = self.app
        in_f, out_f = clip.in_frame, clip.get_out_frame()
        style = app.settings.data.get("trajectory_style", {})

        use_det = None if get_ai_settings(app.settings).get("club_detector", "auto") == "auto" else False

        def worker():
            from ai.club_tracker import track_club
            res = track_club(clip.source_path, in_f, out_f, use_detector=use_det,
                             progress=lambda r, m: self._set_status(f"{m} {r * 100:.0f}%"))
            adir = analysis_dir(app.project_dir, clip.id)
            save_club_track(adir, res, in_f, clip.source_path)
            # 区間ごとに信頼度を判定し、信頼できる区間だけを Swing にする
            seg_q = res.segment_quality()
            good = [(seg, q) for seg, q, ok in seg_q if ok]
            bad = [(seg, q) for seg, q, ok in seg_q if not ok]
            total_span = (res.frames_total - 1)
            # 未描画区間 = 信頼できる区間以外 (区間の隙間 + 信頼度不足の区間)
            drawn = sorted((q["frames"][0] - in_f, q["frames"][1] - in_f) for _, q in good)
            undrawn, cursor = [], 0
            for a, b in drawn:
                if a - cursor >= 2:
                    undrawn.append((cursor, a - 1))
                cursor = b + 1
            if total_span - cursor >= 2:
                undrawn.append((cursor, total_span))
            if not good:
                q = res.quality
                self._undrawn[clip.id] = undrawn
                save_auto_snapshot(adir, [], undrawn)
                app.after(0, lambda: self._show_undrawn(clip.id))
                self._set_status("軌道: 信頼度不足 → 描画しません (全区間)\n"
                                 + " / ".join(q.get("reasons", [])) + "\n"
                                 + f"(観測 {q.get('observed')}/{q.get('frames')}f, 率 {q.get('rate', 0):.0%}, "
                                   f"手元接続 {q.get('attached', 0):.0%}) 手描きで補えます", "#E0704B")
                return
            segs = [[(int(x), int(y), f - in_f) for x, y, f in res.confident_points(points=seg)] for seg, _ in good]
            segs = [sg for sg in segs if len(sg) >= 2]
            self._undrawn[clip.id] = undrawn
            save_auto_snapshot(adir, segs, undrawn)
            app.after(0, lambda: self._apply_trajectory(clip, segs, undrawn, style, res, len(bad)))

        self._start(worker, "軌道生成")

    def _apply_trajectory(self, clip, segs, weak, style, res, n_bad=0):
        """GUI スレッド: 生成した制御点 (区間ごと) を軌道として保存し、編集タブに反映する。"""
        app = self.app
        editing = getattr(app, "_edit_clip_id", None) == clip.id
        if editing:
            swings = app._edit_swings
            edit_in, edit_out = app._edit_in, app._edit_out
            app._edit_push_undo()
        else:
            swings, edit_in, edit_out = app.clip_manager.load_trajectory(clip.id)
        first_idx = None
        for sg in segs:
            new_swing = TrajectoryData(
                points=sg,
                color_start_hex=style.get("color_start_hex", "#FFFF00"),
                color_end_hex=style.get("color_end_hex", "#FF0000"),
                thickness=style.get("thickness", 3), blur=style.get("blur", 0),
                fade_frames=style.get("fade_frames", 0), alpha=style.get("alpha", 0.85),
                handles=[None] * len(sg))
            # 空のスイングがあればそこに入れ、無ければ新しいスイングとして追加 (手描きは消さない)
            idx = next((i for i, s_ in enumerate(swings) if len(s_.points) < 2), None)
            if idx is None:
                swings.append(new_swing)
                idx = len(swings) - 1
            else:
                swings[idx] = new_swing
            if first_idx is None:
                first_idx = idx
        if first_idx is None:
            self._set_status("軌道: 描ける区間がありません", "#E0704B")
            return
        app.clip_manager.save_trajectory(clip.id, swings, edit_in=edit_in, edit_out=edit_out)
        if editing:
            app._edit_swings = swings
            app._edit_spline_cache.clear()
            app._edit_swing_idx = first_idx
            app._edit_update_color_buttons()
            app._edit_update_display()
        app._clips_tab_dirty = app._edit_tab_dirty = True
        q = res.quality
        self._show_undrawn(clip.id)
        msg = (f"完了 ({res.seconds:.0f}秒): 描画 {len(segs)} 区間 / 制御点 {sum(len(s_) for s_ in segs)} 点"
               f" (観測 {q.get('observed')}/{q.get('frames')}f)")
        if n_bad:
            msg += f" / 信頼度不足で非描画 {n_bad} 区間"
        if weak:
            msg += " / 未描画区間は「次の未描画区間へ」で移動し手描きで補える"
        self._set_status(msg, "#E0B040" if (weak or n_bad) else "#88CC88")

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
        cfg = get_ai_settings(app.settings)
        track = load_club_track(analysis_dir(app.project_dir, clip.id))
        unreliable = bool(track) and not track.get("reliable", True)
        segs_q = ((track or {}).get("quality") or {}).get("segments") or []
        partial = bool(track) and not unreliable and any(not sq.get("reliable") for sq in segs_q)
        auto_undrawn = self._undrawn.get(clip.id)
        if auto_undrawn is None:
            try:
                from ai.store import load_clip_ai_settings as _l  # noqa
                import json as _j
                ap = analysis_dir(app.project_dir, clip.id) / "auto_points.json"
                auto_undrawn = [tuple(x) for x in _j.loads(ap.read_text(encoding="utf-8")).get("undrawn", [])] if ap.exists() else []
            except Exception:
                auto_undrawn = []
        partial = partial or bool(auto_undrawn)
        if not swings and not track:
            self._set_status("軌道も骨格もありません。先に自動生成か手描きをしてください", "#E0B040")
            return
        if not swings and unreliable:
            self._set_status("軌道: 信頼度不足のため軌道なしで書き出します (骨格の局面のみ)", "#E0B040")
        camera = self.camera_angle_for(clip.id)
        out_dir = app.project_dir / "exports" / "sns"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip.name}_short.mp4"
        landmarks = landmarks_from_track(track)
        # 信頼度不足の軌道は局面推定にも使わない (骨格由来のみ)
        dense = [tuple(p) for p in track.get("points", [])] if (track and not unreliable) else None
        factor = realtime_factor(clip.fps, cfg["source_kind"])

        def worker():
            from ai.sns_render import render_sns
            rep = render_sns(
                clip.source_path, out_path, in_frame=clip.in_frame, out_frame=clip.get_out_frame(),
                swings=swings, landmarks=landmarks, realtime_factor=factor, options=cfg["sns"],
                dense_points=dense, camera_angle=camera,
                trajectory_note=("軌道: 信頼度不足" if unreliable else ("軌道: 一部区間のみ" if partial else None)),
                progress=lambda r, m: self._set_status(f"{m} {r * 100:.0f}%"))
            ds = self._collect_dataset(clip, swings, "縦型") if swings else None
            dsn = f"  ラベル保存 strong {ds['n_strong']} / weak {ds['n_weak']}" if ds else ""
            self._set_status(f"完了: {out_path.name} ({rep['frames']}f, {rep['seconds']:.0f}秒){dsn}"
                             f"\n{out_dir}", "#88CC88")

        self._start(worker, "縦型書き出し")


def attach_edit_tools(app, parent) -> EditTools:
    return EditTools(app, parent)
