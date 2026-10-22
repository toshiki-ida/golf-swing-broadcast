"""dataset_inventory — datasets/clubhead の棚卸し。学習前に必ず見る。

    python tools/dataset_inventory.py [--dataset <保存先>] [--json <出力先>] [--exclude-holdout]

出すもの:
  - クリップ数、点の総数、由来の内訳 (manual / auto_confirmed / interpolated / negative)
  - クリップごとの点数、撮影方向 (analysis の settings.json / sns json / swing_analyzer の判定が labels.json にあれば)、画像サイズ
  - holdout (tests/real/split.yaml) のクリップが混ざっていないか (元ファイル名で判定)。--exclude-holdout で labels.json に
    "excluded": "holdout" を書いて学習から外す
  - おかしいデータ: 画面外の座標、同一フレームに複数の強ラベル、点が極端に少ないクリップ (強ラベル 3 点未満)、画像ファイル欠落
"""
from __future__ import annotations

import sys as _sys
for _st in (_sys.stdout, _sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.dataset import DATASET_DIR, holdout_stems  # noqa: E402

MIN_STRONG = 3


def camera_of(meta: dict, lab: Path) -> str:
    """撮影方向。labels.json に camera があればそれ、無ければ同じプロジェクトの analysis/<clip>/settings.json・sns json を探す"""
    if meta.get("camera"):
        return str(meta["camera"])
    src = Path(meta.get("source_path", ""))
    # <project>/datasets/clubhead/<clip> → <project>/analysis/<clip>/
    proj = lab.parent.parent.parent.parent
    adir = proj / "analysis" / meta["clip_id"]
    try:
        st = adir / "settings.json"
        if st.exists():
            cam = json.loads(st.read_text(encoding="utf-8")).get("camera_angle")
            if cam and cam != "auto":
                return cam + " (手動)"
        for js in (proj / "exports" / "sns").glob(f"{src.stem}*_short.json"):
            cam = (json.loads(js.read_text(encoding="utf-8")).get("camera") or {}).get("angle")
            if cam:
                return cam
    except Exception:
        pass
    return "unknown"


def main() -> int:
    args = sys.argv[1:]
    root = Path(args[args.index("--dataset") + 1]) if "--dataset" in args else DATASET_DIR
    out_json = Path(args[args.index("--json") + 1]) if "--json" in args else None
    exclude = "--exclude-holdout" in args
    hold = holdout_stems()
    clips, problems = [], []
    totals = Counter()
    sizes = Counter()
    cams = Counter()
    for lab in sorted(root.glob("*/labels.json")):
        meta = json.loads(lab.read_text(encoding="utf-8"))
        pts = meta.get("points", [])
        src = Counter(p["source"] for p in pts)
        totals.update(src)
        stem = Path(meta.get("source_path", "")).stem.replace("_trim", "")
        is_hold = stem in hold or any(stem.startswith(h) for h in hold)
        cam = camera_of(meta, lab)
        cams[cam] += 1
        size = f"{meta.get('width')}x{meta.get('height')}"
        sizes[size] += 1
        W, H = meta.get("width", 0), meta.get("height", 0)
        strong = [p for p in pts if not p.get("weak") and not p.get("negative")]
        # --- 異常検出
        off = [p for p in pts if not (0 <= p["x"] < W and 0 <= p["y"] < H)]
        if off:
            problems.append(f"{meta['clip_id']}: 画面外の座標 {len(off)} 点 (例 {off[0]['x']},{off[0]['y']} F{off[0]['frame']})")
        dup = [f for f, n in Counter(p["frame"] for p in strong).items() if n > 1]
        if dup:
            problems.append(f"{meta['clip_id']}: 同一フレームに複数の強ラベル {sorted(dup)[:6]}")
        if len(strong) < MIN_STRONG:
            problems.append(f"{meta['clip_id']}: 強ラベルが {len(strong)} 点しかない")
        missing = [p["image"] for p in pts if not (lab.parent / p["image"]).exists()]
        if missing:
            problems.append(f"{meta['clip_id']}: 画像が無い {len(missing)} 枚")
        outside = [p for p in pts if not p.get("in_crop", True)]
        if outside:
            problems.append(f"{meta['clip_id']}: 切り出し外の点 {len(outside)} 点 (学習では除外される)")
        if is_hold:
            problems.append(f"{meta['clip_id']}: holdout の素材 ({stem}) → 学習に使ってはいけない" + (" → excluded を記録" if exclude else ""))
            if exclude and meta.get("excluded") != "holdout":
                meta["excluded"] = "holdout"
                lab.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        clips.append({"clip_id": meta["clip_id"], "source": stem, "labeler": meta.get("labeler"), "camera": cam, "size": size,
                      "strong": len(strong), "manual": src.get("manual", 0), "auto_confirmed": src.get("auto_confirmed", 0),
                      "interpolated": src.get("interpolated", 0), "negative": src.get("negative", 0),
                      "holdout": is_hold, "excluded": meta.get("excluded"), "saved": meta.get("saved")})
    n_strong = sum(c["strong"] for c in clips if not c["holdout"])
    print(f"保存先: {root}")
    print(f"クリップ {len(clips)} 本 (holdout {sum(1 for c in clips if c['holdout'])} 本を含む)、点の総数 {sum(totals.values())}")
    print("由来の内訳: " + ", ".join(f"{k} {v}" for k, v in totals.most_common()))
    print(f"学習に使える強ラベル (holdout 除く): {n_strong} 点、負例 {sum(c['negative'] for c in clips if not c['holdout'])} 点")
    print("撮影方向: " + ", ".join(f"{k} {v}" for k, v in cams.most_common()))
    print("画像サイズ: " + ", ".join(f"{k} {v}" for k, v in sizes.most_common()))
    print()
    print(f"{'clip_id':<22}{'素材':<20}{'labeler':<9}{'方向':<16}{'サイズ':<10}{'強':>4}{'手':>4}{'自':>4}{'補':>4}{'負':>4}  備考")
    for c in clips:
        note = ("holdout " if c["holdout"] else "") + (f"excluded={c['excluded']} " if c["excluded"] else "")
        print(f"{c['clip_id']:<22}{c['source']:<20}{str(c['labeler']):<9}{c['camera']:<16}{c['size']:<10}"
              f"{c['strong']:>4}{c['manual']:>4}{c['auto_confirmed']:>4}{c['interpolated']:>4}{c['negative']:>4}  {note}")
    print()
    if problems:
        print("要確認:")
        for p_ in problems:
            print("  - " + p_)
    else:
        print("要確認: なし")
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"root": str(root), "clips": clips, "totals": dict(totals), "cameras": dict(cams),
                                        "sizes": dict(sizes), "problems": problems, "n_strong_usable": n_strong},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
