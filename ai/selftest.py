"""ai.selftest — exe / ソース共通の動作確認: 編集タブの「軌道自動生成 → 縦型動画書き出し」を GUI 上で通す。

    GolfSwingBroadcast.exe --ai-selftest <動画> <出力dir> [--in N] [--out N]
    python app.py           --ai-selftest <動画> <出力dir> [--in N] [--out N]

一時プロジェクト (<出力dir>/project) にクリップを登録して編集タブに読み込み、
ai_edit_tools.run_autotrace() → run_sns_export() を順に実行し、
<出力dir>/selftest_result.json に結果 (OK/ERROR、局面、指標、所要時間) を書いて終了する。
終了コード: 0 = OK, 1 = ERROR。--noconsole の exe でも結果ファイルで確認できる。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path


def run_selftest(app_cls, video: str, out_dir: str, in_f: int = 0, out_f: int = -1) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "selftest_result.json"
    project = out / "project"
    if project.exists():
        shutil.rmtree(project, ignore_errors=True)
    project.mkdir(parents=True)
    res = {"status": "ERROR", "frozen": bool(getattr(sys, "frozen", False)), "video": str(video),
           "python": sys.version.split()[0], "steps": []}
    t0 = time.time()
    try:
        app = app_cls(project_dir=str(project))
    except Exception:
        res["error"] = traceback.format_exc()
        result_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    def fail(msg):
        res["error"] = msg
        res["seconds"] = round(time.time() - t0, 1)
        result_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        app._on_close()

    def start():
        try:
            app._stop_capture()
            clip = app.clip_manager.add_clip(str(video))
            if out_f >= 0 or in_f > 0:
                app.clip_manager.set_in_out(clip.id, in_f, out_f if out_f >= 0 else clip.total_frames - 1)
            app.tabview.set("編集")
            app._edit_select_clip(clip.id)
            res["clip_id"] = clip.id
            res["steps"].append("clip loaded")
            tools = app.ai_edit_tools
            tools.run_autotrace()
            app.after(500, lambda: wait_trace(clip, tools))
        except Exception:
            fail(traceback.format_exc())

    def wait_trace(clip, tools):
        if tools._busy:
            app.after(500, lambda: wait_trace(clip, tools))
            return
        status = tools.status.cget("text")
        res["autotrace_status"] = status
        pts = [len(s.points) for s in app._edit_swings]
        res["swing_points"] = pts
        res["trajectory_gated"] = "信頼度不足" in status
        if res["trajectory_gated"]:
            res["steps"].append("autotrace gated (信頼度不足 → 軌道なしで縦型書き出しへ)")
        elif not pts or max(pts) < 2:
            fail(f"軌道が生成されませんでした: {status}")
            return
        else:
            res["steps"].append("autotrace ok")
        tools.run_sns_export()
        app.after(500, lambda: wait_sns(clip, tools))

    def wait_sns(clip, tools):
        if tools._busy:
            app.after(500, lambda: wait_sns(clip, tools))
            return
        status = tools.status.cget("text")
        res["sns_status"] = status
        mp4 = project / "exports" / "sns" / f"{clip.name}_short.mp4"
        rep = mp4.with_suffix(".json")
        if not mp4.exists() or not rep.exists():
            fail(f"縦型動画が出力されませんでした: {status}")
            return
        r = json.loads(rep.read_text(encoding="utf-8"))
        res["steps"].append("sns ok")
        res["sns_mp4"] = str(mp4)
        res["sns_frames"] = r.get("frames")
        try:
            from ai.club_detector import ClubDetector, model_path
            from ai.dataset import dataset_dir, summary
            res["club_detector"] = {"available": ClubDetector().available(), "model": str(model_path())}
            res["labels"] = {"dir": str(dataset_dir(project)), **summary(dataset_dir(project))}
        except Exception as e:
            res["labels_error"] = str(e)
        res["analysis"] = {k: r.get(k) for k in ("phases", "phase_source", "trajectory_phases",
                                                  "skeleton_phases", "camera", "metrics", "comments",
                                                  "warnings", "tempo", "layout", "trajectory_drawn", "trajectory_note")}
        res["status"] = "OK"
        res["seconds"] = round(time.time() - t0, 1)
        result_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        app._on_close()

    app.after(1500, start)
    app.mainloop()
    return 0 if res["status"] == "OK" else 1
