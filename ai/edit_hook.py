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
from ai.trajectory_prior import (fill_trajectory, control_points, load_prior, swing_frame_range,
                                 estimate_camera_from_landmarks)
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
        # 信頼度ゲート: 閾値と各区間の指標 (どの区間がなぜ止まったか)
        self.gate_label = ctk.CTkLabel(sec, text="", font=("Consolas", 10), text_color="#BBBBBB",
                                       wraplength=240, justify="left")
        self.gate_label.pack(anchor="w", padx=8, pady=(2, 0))
        self.draw_all_btn = ctk.CTkButton(sec, text="全区間を描く (ゲート無視)", height=24, font=("", 11),
                                          fg_color="#4A4A2A", hover_color="#6A6A3A", command=self.draw_all_segments)
        self.draw_all_btn.pack(fill="x", padx=8, pady=(2, 0))
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
        self._estimated: dict = {}    # clip_id -> [(a, b), ...] 推定で埋めた区間
        self._undrawn_idx = 0
        self._wrap_save_trajectory()
        self._wrap_edit_export()
        self._wrap_select_clip()
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
                print(f"[AI] ラベル収集: {clip.name} → strong {res['n_strong']} / weak {res['n_weak']} / "
                      f"negative {res.get('n_negative', 0)} {res['sources']} → {res['dir']}")
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

    def _wrap_select_clip(self):
        """クリップ選択時に、保存済みの区間指標・未描画区間・撮影方向を右パネルへ反映する"""
        app = self.app
        if getattr(app, "_ai_select_wrapped", False) or not hasattr(app, "_edit_select_clip"):
            return
        orig = app._edit_select_clip

        def wrapped(clip_id, _orig=orig):
            _orig(clip_id)
            try:
                if clip_id not in self._undrawn:
                    import json as _j
                    ap = analysis_dir(app.project_dir, clip_id) / "auto_points.json"
                    if ap.exists():
                        auto = _j.loads(ap.read_text(encoding="utf-8"))
                        self._undrawn[clip_id] = [tuple(u) for u in auto.get("undrawn", [])]
                        spans, cur = [], None
                        for f in sorted(auto.get("estimated", [])):
                            if cur and f == cur[1] + 1:
                                cur[1] = f
                            else:
                                cur = [f, f]
                                spans.append(cur)
                        self._estimated[clip_id] = [(a, b) for a, b in spans]
                self._show_undrawn(clip_id)
                if hasattr(self, "camera_seg"):
                    self.camera_seg.set(self.camera_angle_for(clip_id))
            except Exception as e:
                print(f"[AI] 右パネル更新に失敗: {e}")

        app._edit_select_clip = wrapped
        app._ai_select_wrapped = True

    def _show_gate(self, clip_id):
        """信頼度ゲートの閾値と、各区間の指標・判定を右パネルに出す (club_track.json から)"""
        try:
            track = load_club_track(analysis_dir(self.app.project_dir, clip_id)) if clip_id else None
        except Exception:
            track = None
        q = (track or {}).get("quality") or {}
        gate = q.get("gate") or get_ai_settings(self.app.settings).get("gate", {})
        lines = [f"ゲート: 観測≥{gate.get('min_observed', '?')}f 率≥{gate.get('min_rate', 0):.0%} "
                 f"連続≥{gate.get('min_continuity', 0):.0%} 接続≥{gate.get('min_hand_connect_ratio', 0):.0%} "
                 f"動き≥{gate.get('min_motion_bh', 0):.2f} 頭上≤{gate.get('max_on_head_ratio', 1):.0%} "
                 f"距σ≤{gate.get('max_dist_std', 9):.2f} 距Δ≤{gate.get('max_dist_change', 9):.3f} "
                 f"速比≥{gate.get('min_speed_ratio', 0):.1f} 手≥{gate.get('min_wrist_ratio', 0):.0%}"]
        in_f = int((track or {}).get("in_frame", 0))
        for i, sg in enumerate(q.get("segments") or []):
            a, b = sg.get("frames", (0, 0))
            a, b = int(a) - in_f, int(b) - in_f
            vals = (f"率{sg.get('rate', 0):.0%} 連続{sg.get('continuity', 0):.0%} 接続{sg.get('attached', 0):.0%} "
                    f"動き{sg.get('motion_bh', 0):.2f} 頭上{sg.get('on_head', 0):.0%} 距σ{sg.get('dist_std', 0):.2f} "
                    f"距Δ{sg.get('dist_change', 0):.3f} 速比{min(sg.get('speed_ratio', 0), 99):.1f} 手{sg.get('wrist_motion', 0):.2f}")
            if sg.get("reliable"):
                lines.append(f"区間{i + 1} F{a}-{b} 描画 | {vals}")
            else:
                lines.append(f"区間{i + 1} F{a}-{b} 未描画 | {vals} ← " + " / ".join(sg.get("reasons") or []))
        if not q.get("segments"):
            lines.append("(自動生成の区間なし)" if track else "(自動生成の結果なし)")
        self.gate_label.configure(text="\n".join(lines))

    def draw_all_segments(self):
        """ゲートで止めた区間も含めて、自動生成の全区間を描く (Undo 可)。学習データには通常どおり承認として残る"""
        clip = self._current_clip()
        if clip is None:
            return
        import json as _j
        ap = analysis_dir(self.app.project_dir, clip.id) / "auto_points.json"
        if not ap.exists():
            self._set_status("自動生成の結果がありません。先に「ヘッド軌道を自動生成」", "#E0B040")
            return
        auto = _j.loads(ap.read_text(encoding="utf-8"))
        segs = [[tuple(p) for p in sg] for sg in auto.get("all_segments") or auto.get("segments") or []]
        segs = [sg for sg in segs if len(sg) >= 2]
        if not segs:
            self._set_status("描ける区間がありません (観測なし)", "#E0B040")
            return
        style = self.app.settings.data.get("trajectory_style", {})
        total = max(0, getattr(self.app, "_edit_total", 0) - 1)
        drawn = sorted((sg[0][2], sg[-1][2]) for sg in segs)
        undrawn, cursor = [], 0
        for a, b in drawn:
            if a - cursor >= 2:
                undrawn.append((cursor, a - 1))
            cursor = b + 1
        if total - cursor >= 2:
            undrawn.append((cursor, total))
        self._undrawn[clip.id] = undrawn
        drawn_sets = [set(tuple(p) for p in sg) for sg in auto.get("segments", [])]
        # ゲートで止めていた区間は extra_segments に記録 (corrections では「残っていれば自動点」として扱う)
        auto["extra_segments"] = [[list(p) for p in sg] for sg in segs if set(sg) not in drawn_sets]
        auto["undrawn"] = [list(u) for u in undrawn]
        auto["gate_ignored"] = True
        ap.write_text(_j.dumps(auto, ensure_ascii=False), encoding="utf-8")
        # 既に描いてある自動生成のスイング (手直し前のもの) は置き換える。手描きの点は残す
        app = self.app
        if getattr(app, "_edit_clip_id", None) == clip.id:
            app._edit_push_undo()
            keep = [sw for sw in app._edit_swings if set(tuple(p) for p in sw.points) not in drawn_sets]
            app._edit_swings = keep or [app._make_default_trajectory(0)]
            app._edit_swing_idx = 0
        self._apply_trajectory(clip, segs, undrawn, style, None, 0)
        self._set_status(f"全 {len(segs)} 区間を描きました (ゲート無視)。確認して不要な区間は Delete で消してください", "#E0B040")

    def _show_undrawn(self, clip_id):
        self._show_gate(clip_id)
        spans = self._undrawn.get(clip_id) or []
        est = self._estimated.get(clip_id) or []
        if not spans and est:
            txt = "推定で補った区間 (中抜きの点。確認して直す): " + " ".join(f"F{a}-{b}" for a, b in est[:6])
            if len(est) > 6:
                txt += f" 他 {len(est) - 6}"
            self.undrawn_label.configure(text=txt, text_color="#9ECBFF")
            return
        self.undrawn_label.configure(text_color="#E0B040")
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
        spans = self._undrawn.get(cid) or self._estimated.get(cid) or []
        if not spans:
            self._set_status("未描画・推定の区間はありません")
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
        src_of = {}
        for sw in app._edit_swings:
            srcs = getattr(sw, "sources", None) or []
            for i, p in enumerate(sw.points):
                src_of.setdefault(tuple(p), srcs[i] if i < len(srcs) else "manual")
        pts = sorted({tuple(p) for sw in app._edit_swings for p in sw.points}, key=lambda q: q[2])
        base = app._edit_swings[0]
        merged = TrajectoryData(points=pts, color_start_hex=base.color_start_hex, color_end_hex=base.color_end_hex,
                                thickness=base.thickness, blur=getattr(base, "blur", 0),
                                fade_frames=getattr(base, "fade_frames", 0), alpha=getattr(base, "alpha", 0.85),
                                handles=[None] * len(pts), sources=[src_of.get(p, "manual") for p in pts])
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

        cfg = get_ai_settings(app.settings)
        use_det = None if cfg.get("club_detector", "auto") == "auto" else False
        gate = cfg.get("gate")
        use_fill = cfg.get("trajectory_fill", "prior") == "prior"
        cam_setting = self.camera_angle_for(clip.id)

        def worker():
            from ai.club_tracker import track_club
            res = track_club(clip.source_path, in_f, out_f, use_detector=use_det, gate=gate,
                             progress=lambda r, m: self._set_status(f"{m} {r * 100:.0f}%"))
            adir = analysis_dir(app.project_dir, clip.id)
            save_club_track(adir, res, in_f, clip.source_path)
            # 区間ごとに信頼度を判定し、信頼できる区間だけを Swing にする
            seg_q = res.segment_quality()
            all_segs = [[(int(x), int(y), f - in_f) for x, y, f in res.confident_points(points=seg)] for seg, _, _ in seg_q]
            seg_quality = [{**q, "reliable": ok, "frames": [q["frames"][0] - in_f, q["frames"][1] - in_f]}
                           for _, q, ok in seg_q]
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
                save_auto_snapshot(adir, [], undrawn, all_segments=all_segs, segment_quality=seg_quality)
                app.after(0, lambda: self._show_undrawn(clip.id))
                self._set_status("軌道: 信頼度ゲートで全区間が止まりました → 自動では描きません\n"
                                 + " / ".join(q.get("reasons", [])) + "\n"
                                 + f"(観測 {q.get('observed')}/{q.get('frames')}f, 率 {q.get('rate', 0):.0%}, "
                                   f"手元接続 {q.get('attached', 0):.0%}) 下の区間一覧で理由を確認。"
                                   "「全区間を描く」か手描きで補えます", "#E0704B")
                return
            segs = [[(int(x), int(y), f - in_f) for x, y, f in res.confident_points(points=seg)] for seg, _ in good]
            segs = [sg for sg in segs if len(sg) >= 2]
            sources = None
            est_frames = []
            if use_fill:
                try:
                    filled_segs, sources, est_frames, est_spans, fill_info = self._fill_with_prior(
                        res, in_f, clip, seg_q, segs, cam_setting)
                    if filled_segs:
                        segs = filled_segs
                        undrawn = []
                        self._estimated[clip.id] = est_spans
                        print(f"[AI] 推定で補完: {fill_info}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[AI] 推定補完に失敗 ({e})。観測区間だけ描きます")
            self._undrawn[clip.id] = undrawn
            save_auto_snapshot(adir, segs, undrawn, all_segments=all_segs, segment_quality=seg_quality,
                               estimated=est_frames)
            app.after(0, lambda: self._apply_trajectory(clip, segs, undrawn, style, res, len(bad), sources=sources))

        self._start(worker, "軌道生成")

    def _fill_with_prior(self, res, in_f, clip, seg_q, segs, cam_setting):
        """観測 (信頼できる区間) の無いフレームを推定で埋め、1 本の制御点列にする。
        戻り値: (segs, sources, estimated_frames, estimated_spans, info)。推定できなければ ([], None, [], [], reason)"""
        W, H = int(getattr(clip, "width", 1920) or 1920), int(getattr(clip, "height", 1080) or 1080)
        observed = {f - in_f: (x, y) for seg, q, ok in seg_q if ok for x, y, f in seg}
        wrist = {f - in_f: w for f, w in (res.wrist or {}).items()}
        lm_rel = {f - in_f: lm for f, lm in (getattr(res, "landmarks", {}) or {}).items()}
        angle = cam_setting if cam_setting in ("dtl", "face_on") else estimate_camera_from_landmarks(lm_rel, W, H, res.real_fps)
        prior = load_prior(angle) if angle in ("dtl", "face_on") else None
        rng = swing_frame_range(lm_rel, W, H, res.real_fps)
        if rng is None and observed:
            rng = (min(observed), max(observed))
        if rng is None:
            return [], None, [], [], "骨格の局面も観測も無い"
        if observed:
            rng = (min(rng[0], min(observed)), max(rng[1], max(observed)))
        frames = list(range(int(rng[0]), int(rng[1]) + 1))
        filled = fill_trajectory(observed, wrist, res.body_h, res.real_fps, frames, prior=prior)
        if not filled or not any(s == "estimated" for _, _, _, s in filled):
            return [], None, [], [], "推定が必要な区間なし" if filled else "手首が取れない"
        obs_ctrl = [(x, y, f) for sg in segs for x, y, f in sg]
        ctrl = control_points(filled, res.real_fps, observed_ctrl=obs_ctrl)
        if len(ctrl) < 2:
            return [], None, [], [], "制御点が足りない"
        est_frames = [f for x, y, f, s in filled if s == "estimated"]
        spans, cur = [], None
        for f in sorted(est_frames):
            if cur and f == cur[1] + 1:
                cur[1] = f
            else:
                cur = [f, f]
                spans.append(cur)
        spans = [(a, b) for a, b in spans]
        info = (f"{angle} prior={'あり' if prior else 'なし(物理のみ)'} 観測 {sum(1 for p in filled if p[3] == 'observed')}f / "
                f"推定 {len(est_frames)}f 区間 {spans[:5]}")
        return [[(x, y, f) for x, y, f, s in ctrl]], [[s for x, y, f, s in ctrl]], est_frames, spans, info

    def _apply_trajectory(self, clip, segs, weak, style, res, n_bad=0, sources=None):
        """GUI スレッド: 生成した制御点 (区間ごと) を軌道として保存し、編集タブに反映する。"""
        app = self.app
        editing = getattr(app, "_edit_clip_id", None) == clip.id
        if editing:
            swings = app._edit_swings
            edit_in, edit_out = app._edit_in, app._edit_out
            if res is not None:          # draw_all_segments は呼ぶ前に push 済み
                app._edit_push_undo()
        else:
            swings, edit_in, edit_out = app.clip_manager.load_trajectory(clip.id)
        first_idx = None
        for si_, sg in enumerate(segs):
            src = list(sources[si_]) if sources and si_ < len(sources) else ["observed"] * len(sg)
            new_swing = TrajectoryData(
                points=sg,
                color_start_hex=style.get("color_start_hex", "#FFFF00"),
                color_end_hex=style.get("color_end_hex", "#FF0000"),
                thickness=style.get("thickness", 3), blur=style.get("blur", 0),
                fade_frames=style.get("fade_frames", 0), alpha=style.get("alpha", 0.85),
                handles=[None] * len(sg), sources=src)
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
        self._show_undrawn(clip.id)
        if res is None:
            return
        q = res.quality
        n_est = sum(1 for s_ in (sources or []) for v in s_ if v == "estimated")
        msg = (f"完了 ({res.seconds:.0f}秒): 描画 {len(segs)} 区間 / 制御点 {sum(len(s_) for s_ in segs)} 点"
               + (f" (うち推定 {n_est} 点)" if n_est else "")
               + f" (観測 {q.get('observed')}/{q.get('frames')}f)")
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
            self._set_status("軌道なしで書き出します (骨格の局面のみ)。動画に注記は入りません", "#E0B040")
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
                trajectory_note=None,   # 動画には注記を入れない (状態は右パネルのみ)
                progress=lambda r, m: self._set_status(f"{m} {r * 100:.0f}%"))
            ds = self._collect_dataset(clip, swings, "縦型") if swings else None
            dsn = f"  ラベル保存 strong {ds['n_strong']} / weak {ds['n_weak']} / neg {ds.get('n_negative', 0)}" if ds else ""
            self._set_status(f"完了: {out_path.name} ({rep['frames']}f, {rep['seconds']:.0f}秒){dsn}"
                             f"\n{out_dir}", "#88CC88")

        self._start(worker, "縦型書き出し")


def attach_edit_tools(app, parent) -> EditTools:
    return EditTools(app, parent)
