"""eval_club — クラブヘッド軌道の精度評価(正解ラベルとの誤差を手法ごとに一覧)。

tests/club_labels.json (tools/label_club.py で作成) と、各手法が出力した予測 JSON を比較する。

使い方:
    python tools/eval_club.py <予測JSON ...>
    python tools/eval_club.py build/club_eval/*/*.pred.json
    python tools/eval_club.py --all --split tune|holdout|all [--tag before|after_N|<任意>] [--no-run] [--interlaced]
                              [--include-claude-labels]

--all: tests/real/trimmed/ (無ければ tests/real/) の動画を ai/club_tracker.py で追跡し
       (build/club_eval/real/<名前>/)、追跡率・局面を全素材で集計、ラベルのある素材は誤差も評価して
       tests/real/results/<日時>_<tag>_<split>_<commit>.md に残す (既存ファイルは上書きしない)。
       --split: tests/real/split.yaml の tune / holdout を選ぶ (省略時 tune。holdout は修正完了後のみ実行する)。
       正解ラベルは "labeler": "owner" のものだけを使う。Claude の目視ラベルは --include-claude-labels で参考表示。
       --no-run は追跡を省略して既存の予測で評価だけ行う。
       --interlaced は 1080i 素材をフィールド単位で処理する (ファイル名に "_i" か
       ラベルに "interlaced": true があれば自動で付く)。

予測 JSON の形式 (手法ごとに 1 ファイル):
    {
      "method": "autotrace",            # 手法名
      "video": "sample_30fps.mp4",      # tests/club_labels.json のキー (ファイル名)
      "points": [[x, y, frame], ...],   # 全フレームの推定位置 (フレームは動画 index、float 可)
      "conf": {"63": 2.1, ...},         # 任意: フレームごとの信頼度
      "seconds": 9.3,                   # 任意: 処理時間
      "note": "GPU RTX3050"             # 任意
    }

評価:
    - 正解フレームに予測点があればその誤差 (px)、無ければ前後の点から線形補間して誤差を出し "*" を付ける
    - 手法ごとに 平均 / 中央値 / 最大 誤差、評価できた点数、処理時間を表示
    - 誤差は元解像度のピクセル。画面高さ比 (%) も併記する
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # ai / clip_manager を import するため
LABELS_PATH = ROOT / "tests" / "club_labels.json"
PHASE_ORDER = ["address", "takeaway", "top", "downswing", "impact", "follow"]


def predict_at(points: list, frame: float):
    """frame での予測位置を返す。(x, y, interpolated)。予測範囲外なら None。"""
    if not points:
        return None
    pts = sorted(points, key=lambda p: p[2])
    fs = np.array([p[2] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    exact = np.flatnonzero(np.abs(fs - frame) < 1e-6)
    if len(exact):
        i = exact[0]
        return float(xs[i]), float(ys[i]), False
    if frame < fs[0] or frame > fs[-1]:
        return None
    return float(np.interp(frame, fs, xs)), float(np.interp(frame, fs, ys)), True


REAL_DIR = ROOT / "tests" / "real"
TRIM_DIR = REAL_DIR / "trimmed"
SPLIT_PATH = REAL_DIR / "split.yaml"
VIDEO_EXTS = (".mp4", ".mov", ".mxf", ".mkv", ".avi")


def track_summary(video: Path, out_dir: Path) -> dict:
    """1 本分の追跡率と局面検出 (軌道由来 / 骨格由来 / 統合) をまとめる。club_tracker や analyzer は変更せず呼ぶだけ。"""
    import logging
    row = {"name": video.name}
    pred_path = out_dir / "club_tracker.pred.json"
    track_path = out_dir / "club_track.json"
    if not pred_path.exists():
        row["error"] = "追跡失敗 (出力なし)"
        return row
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    conf = pred.get("conf", {})
    row.update({"frames": len(pred.get("points", [])), "observed": len(pred.get("raw", {})),
                "confident": sum(1 for v in conf.values() if v >= 1.5),
                "weak": len(pred.get("weak_ranges", [])), "seconds": pred.get("seconds")})
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
        cap.release()
        row["video_frames"] = n
        from clip_manager import TrajectoryData
        from ai.config import realtime_factor
        from ai.sns_render import derive_phases, fuse_phases
        from ai.store import landmarks_from_track
        from ai.swing_analyzer import analyze
        track = json.loads(track_path.read_text(encoding="utf-8")) if track_path.exists() else None
        dense = [tuple(p) for p in (track or pred)["points"]]
        swings = [TrajectoryData(points=[tuple(p) for p in pred["confident_points"]])]
        traj = derive_phases(swings, 0, dense)
        row["traj_phases"] = traj
        lm = landmarks_from_track(track)
        warnings = []
        if lm:
            factor = realtime_factor(fps, "auto")
            logging.getLogger("swing_analyzer").setLevel(logging.ERROR)
            logging.getLogger("sns_render").setLevel(logging.ERROR)
            a = analyze(lm, W, H, fps, fps * factor, video_path=str(video), in_frame=0, dense_points=dense)
            row["skel_phases"] = a["phases"]
            row["camera"] = f"{a['camera']['angle']} ({a['camera']['ratio']:.2f}, {a['camera']['confidence']:.2f})"
            row["impact_method"] = a["impact"]["chosen"]
            row["swing_frames"] = a["phases"]["finish"] - a["phases"]["address"]
            warnings += a["warnings"]
            fused, src = fuse_phases(traj, a["phases"], warnings, fps * factor)
            row["fused"] = fused
            row["fused_src"] = src
            tol = max(2, int(round(0.05 * fps * factor)))   # 実時間 50 ms
            row["tol"] = tol
            row["agree"] = {k: abs(traj[k] - a["phases"][k]) <= tol for k in ("top", "impact")}
        row["warnings"] = warnings
    except Exception as e:
        row["analyze_error"] = f"{type(e).__name__}: {e}"
    return row


def format_summary(rows: list) -> str:
    lines = ["=== 追跡率・局面検出 (全素材) ===",
             f"{'素材':<22} {'動画f':>6} {'観測':>6} {'確信':>6} {'率':>6} {'欠測':>5} {'秒':>5}  "
             f"{'軌道 top/imp':<13} {'骨格 top/imp':<13} {'採用':<10} {'一致':<8} {'ｽｲﾝｸﾞf':>7} {'撮影方向 (比, 信頼度)':<24} 警告"]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['name']:<22} {r['error']}")
            continue
        rate = f"{100 * r['observed'] / max(1, r.get('video_frames', r['frames'])):.0f}%"
        tp = r.get("traj_phases") or {}
        sp = r.get("skel_phases") or {}
        src = r.get("fused_src") or {}
        warn = "; ".join(r.get("warnings", []))
        if r.get("analyze_error"):
            warn = (warn + "; " if warn else "") + r["analyze_error"]
        tps = f"{tp.get('top', '-')}/{tp.get('impact', '-')}" if tp else "-"
        sps = f"{sp.get('top', '-')}/{sp.get('impact', '-')}" if sp else "-"
        srcs = f"{str(src.get('top', '-'))[:4]}/{str(src.get('impact', '-'))[:4]}" if src else "-"
        ag = r.get("agree")
        ags = ("".join("o" if ag[k] else "x" for k in ("top", "impact")) + f"(±{r.get('tol')})") if ag else "-"
        lines.append(f"{r['name']:<22} {r.get('video_frames', '?'):>6} {r['observed']:>6} {r['confident']:>6} {rate:>6} {r['weak']:>5} "
                     f"{(r.get('seconds') or 0):>5.0f}  {tps:<13} {sps:<13} {srcs:<10} {ags:<8} {str(r.get('swing_frames', '-')):>7} "
                     f"{r.get('camera', '-'):<24} {warn}")
    lines.append("観測 = 位置を観測できたフレーム数、確信 = 信頼度 1.5 以上、率 = 観測/動画フレーム、欠測 = 欠測区間の数。"
                 "採用 = 統合後の出所 (traj=軌道 / skel=骨格)。一致 = トップ/インパクトで軌道と骨格が実時間 50ms 以内か (o/x)。ｽｲﾝｸﾞf = 骨格由来のアドレス→フィニッシュのフレーム数。")
    return chr(10).join(lines)


def run_all(args: list) -> int:
    """tests/real/ の素材を一括で追跡・評価し、結果を tests/real/results/ に残す。"""
    import datetime
    import io
    import subprocess
    from contextlib import redirect_stdout

    tag = args[args.index("--tag") + 1] if "--tag" in args else "run"
    no_run = "--no-run" in args
    force_i = "--interlaced" in args
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8")) if LABELS_PATH.exists() else {}
    include_claude = "--include-claude-labels" in args
    if not include_claude:   # 正解はオーナーのラベルだけ。Claude の目視ラベルは参考扱い
        labels = {k: v for k, v in labels.items() if v.get("labeler") == "owner"}
    split = args[args.index("--split") + 1] if "--split" in args else "tune"
    src_dir = TRIM_DIR if TRIM_DIR.is_dir() and any(TRIM_DIR.glob("*_trim.*")) else REAL_DIR
    videos = sorted(p for p in src_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if SPLIT_PATH.exists() and split != "all":
        import yaml
        sp = yaml.safe_load(SPLIT_PATH.read_text(encoding="utf-8")) or {}
        chosen = sp.get(split) or {}
        if not chosen:
            print(f"split.yaml に {split} がありません")
            return 1
        stems = {Path(k).stem for k in chosen}
        videos = [v for v in videos if v.stem.replace("_trim", "") in stems]
        if split == "holdout":
            print("*** holdout を実行します。修正中に実行してはいけません ***")
    if not videos:
        print(f"素材がありません: {src_dir} (tools/trim_real.py で切り出し、split.yaml を確認)")
        return 1
    print(f"対象: {len(videos)} 本 ({src_dir.name}, split={split}, ラベル {len(labels)} 件)")
    preds = []
    summary_rows = []   # 全素材の追跡率・局面・警告 (ラベルの有無によらず)
    for v in videos:
        out_dir = ROOT / "build" / "club_eval" / "real" / v.stem
        pred = out_dir / "club_tracker.pred.json"
        lab = labels.get(v.name, {})
        inter = force_i or "_i" in v.stem.lower() or bool(lab.get("interlaced"))
        if not no_run or not pred.exists():
            cmd = [sys.executable, str(ROOT / "tools" / "run_club_tracker.py"), str(v), str(out_dir)]
            if inter:
                cmd.append("--interlaced")
            print(f"[run ] {v.name}{' (interlaced)' if inter else ''}", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith("[club_tracker]")]
            print("       " + (tail[-2] if len(tail) >= 2 else (r.stdout[-300:] + r.stderr[-300:]).strip()), flush=True)
        summary_rows.append(track_summary(v, out_dir))
        if v.name in labels and pred.exists():
            preds.append(str(pred))
        elif v.name not in labels:
            print(f"       ラベルなし → 追跡率と確認動画のみ ({out_dir / 'club_tracker_overlay.mp4'})", flush=True)
    summary_text = format_summary(summary_rows)
    print(summary_text)
    if not preds:
        text = "(正解ラベルのある素材がないため誤差評価なし)"
        rc = 0
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = evaluate(preds)
        text = buf.getvalue()
        print(text)
    text = summary_text + chr(10) + chr(10) + text
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip() or "nogit"
        dirty = "+dirty" if subprocess.run(["git", "status", "--porcelain", "ai/", "tools/"],
                                           capture_output=True, text=True, cwd=ROOT).stdout.strip() else ""
    except Exception:
        commit, dirty = "nogit", ""
    res_dir = REAL_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = res_dir / f"{stamp}_{tag}_{split}_{commit}{dirty}.md"
    header = f"# club_tracker 評価 ({stamp}, tag={tag}, split={split}, src={src_dir.name}, commit={commit}{dirty})"
    body = f"素材: {len(summary_rows)} 本 (正解ラベルあり {len(preds)} 本)"
    out.write_text(header + "\n\n" + body + "\n\n```\n" + text + "\n```\n",
                   encoding="utf-8")
    print(f"結果を保存: {out}")
    return rc


def main() -> int:
    if "--all" in sys.argv[1:]:
        return run_all(sys.argv[1:])
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths or not LABELS_PATH.exists():
        print(__doc__)
        if not LABELS_PATH.exists():
            print(f"ラベルがありません: {LABELS_PATH} (先に tools/label_club.py)")
        return 1
    return evaluate([str(p) for p in paths])


def evaluate(pred_paths: list) -> int:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    paths = [Path(p) for p in pred_paths]
    preds = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["video"] not in labels:
            print(f"[skip] {p.name}: ラベルなし ({d['video']})")
            continue
        preds.append(d)
    if not preds:
        return 1

    for video in sorted({d["video"] for d in preds}):
        summary = defaultdict(list)  # method -> [err_px] (動画ごと)
        lab = labels[video]
        H = lab["height"]
        methods = [d for d in preds if d["video"] == video]
        names = [d["method"] for d in methods]
        print(f"\n=== {video}  ({lab['width']}x{H}, 正解 {len(lab['frames'])} 点) ===")
        head = f"{'frame':>5} {'phase':<10} {'正解':<12}" + "".join(f"{n:<22}" for n in names)
        print(head)
        print("-" * len(head))
        frames = sorted(lab["frames"].items(),
                        key=lambda kv: (PHASE_ORDER.index(kv[1].get("phase", "follow"))
                                        if kv[1].get("phase") in PHASE_ORDER else 9, int(kv[0])))
        for f, l in frames:
            line = f"{int(f):>5} {l.get('phase', ''):<10} {'(%d,%d)' % (l['x'], l['y']):<12}"
            for d in methods:
                r = predict_at(d["points"], float(f))
                if r is None:
                    line += f"{'-':<22}"
                    continue
                px, py, interp = r
                err = float(np.hypot(px - l["x"], py - l["y"]))
                summary[d["method"]].append((err, err / H * 100))
                line += f"{'%5.1f px (%.1f%%)%s' % (err, err / H * 100, '*' if interp else ''):<22}"
            print(line)
        print("-" * len(head))
        for d in methods:
            errs = [e for e, _ in summary[d["method"]]]
            sec = d.get("seconds")
            conf = d.get("conf") or {}
            n_conf = sum(1 for v in conf.values() if v >= 1.5) if conf else None
            info = f"{d['method']:<12}: "
            if errs:
                info += (f"平均 {np.mean(errs):6.1f} px  中央 {np.median(errs):6.1f} px  "
                         f"最大 {np.max(errs):6.1f} px  (n={len(errs)})")
            else:
                info += "評価点なし"
            if sec is not None:
                info += f"  処理 {sec:.1f}s"
            if n_conf is not None:
                info += f"  確信点 {n_conf}/{len(conf)}"
            if d.get("note"):
                info += f"  [{d['note']}]"
            print(info)
    print("\n* = 前後の推定点から線形補間した位置との誤差")
    return 0


if __name__ == "__main__":
    sys.exit(main())
