"""eval_club — クラブヘッド軌道の精度評価(正解ラベルとの誤差を手法ごとに一覧)。

tests/club_labels.json (tools/label_club.py で作成) と、各手法が出力した予測 JSON を比較する。

使い方:
    python tools/eval_club.py <予測JSON ...>
    python tools/eval_club.py build/club_eval/*/*.pred.json
    python tools/eval_club.py --all [--tag before|after|<任意>] [--no-run] [--interlaced]

--all: tests/real/ にある動画のうち tests/club_labels.json にラベルがあるものを
       すべて ai/club_tracker.py で追跡し (build/club_eval/real/<名前>/)、評価して
       tests/real/results/<日時>_<tag>_<commit>.md に結果を残す (既存ファイルは上書きしない)。
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
VIDEO_EXTS = (".mp4", ".mov", ".mxf", ".mkv", ".avi")


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
    videos = sorted(p for p in REAL_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS) if REAL_DIR.exists() else []
    if not videos:
        print(f"素材がありません: {REAL_DIR} に動画を置き、tools/label_club.py でラベルを付けてください")
        return 1
    preds = []
    for v in videos:
        if v.name not in labels:
            print(f"[skip] {v.name}: ラベルなし (tools/label_club.py {v} --frames ...)")
            continue
        out_dir = ROOT / "build" / "club_eval" / "real" / v.stem
        pred = out_dir / "club_tracker.pred.json"
        inter = force_i or "_i" in v.stem.lower() or bool(labels[v.name].get("interlaced"))
        if not no_run or not pred.exists():
            cmd = [sys.executable, str(ROOT / "tools" / "run_club_tracker.py"), str(v), str(out_dir)]
            if inter:
                cmd.append("--interlaced")
            print(f"[run ] {v.name}{' (interlaced)' if inter else ''}")
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith("[club_tracker]")]
            print("       " + (tail[-2] if len(tail) >= 2 else r.stdout[-300:] + r.stderr[-300:]))
        if pred.exists():
            preds.append(str(pred))
    if not preds:
        return 1
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = evaluate(preds)
    text = buf.getvalue()
    print(text)
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
    out = res_dir / f"{stamp}_{tag}_{commit}{dirty}.md"
    header = f"# club_tracker 評価 ({stamp}, tag={tag}, commit={commit}{dirty})"
    body = "素材: " + ", ".join(Path(p).parent.name for p in preds)
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
