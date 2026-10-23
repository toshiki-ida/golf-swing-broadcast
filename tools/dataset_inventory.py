"""dataset_inventory — datasets/clubhead の棚卸し。学習前に必ず見る。

    python tools/dataset_inventory.py [--dataset <保存先>] [--json <出力先>] [--exclude-holdout]
                                      [--previous <前回の inventory.json>] [--estimate-camera]

  --previous        前回の棚卸し (json) と比べ、増分 (新しいクリップ・点数・由来・撮影方向) を分けて表示する
  --estimate-camera camera が無いクリップについて、切り出し画像に MediaPipe Pose を掛けて肩幅/胴長の比から
                    撮影方向 (face_on / dtl / unknown) を推定し labels.json に書く (手動の値は上書きしない)

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


def estimate_camera(lab: Path, meta: dict, n_samples: int = 6):
    """切り出し画像 (手首中心) に MediaPipe Pose を掛け、肩幅 / 胴長 の比で撮影方向を推定する。
    swing_analyzer と同じ閾値: < 0.35 → dtl (後方)、> 0.55 → face_on (正面)、間は不明。多数決。"""
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from ai import default_pose_model
    except Exception as e:  # pragma: no cover
        return "unknown", f"mediapipe なし ({e})"
    model = default_pose_model()
    if not Path(model).exists():
        return "unknown", "pose model なし"
    opts = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                                        running_mode=vision.RunningMode.IMAGE, num_poses=3, min_pose_detection_confidence=0.3)
    lm = vision.PoseLandmarker.create_from_options(opts)
    strong = [p for p in meta.get("points", []) if not p.get("weak") and not p.get("negative") and p.get("in_crop", True)
              and (lab.parent / p["image"]).exists()]
    if not strong:
        lm.close()
        return "unknown", "強ラベルなし"
    # スイング中は肩が回って比が変わるので、アドレス付近 (最初のフレーム) だけを使う
    strong = sorted(strong, key=lambda q: q["frame"])[:n_samples]
    votes = {"face_on": 0, "dtl": 0, "unknown": 0}
    ratios = []
    for p in strong:
        im = cv2.imread(str(lab.parent / p["image"]))
        if im is None:
            continue
        H, W = im.shape[:2]
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        best = None
        for pose in res.pose_landmarks or []:
            pts = [(q.x * W, q.y * H) for q in pose]
            # 切り出しの中心 (手首) に最も近い人物
            d = abs(pts[15][0] + pts[16][0] - W) + abs(pts[15][1] + pts[16][1] - H)
            if best is None or d < best[0]:
                best = (d, pts)
        if best is None:
            continue
        pts = best[1]
        sh = ((pts[11][0] - pts[12][0]) ** 2 + (pts[11][1] - pts[12][1]) ** 2) ** 0.5
        hip = ((pts[23][0] + pts[24][0]) / 2, (pts[23][1] + pts[24][1]) / 2)
        shc = ((pts[11][0] + pts[12][0]) / 2, (pts[11][1] + pts[12][1]) / 2)
        torso = ((shc[0] - hip[0]) ** 2 + (shc[1] - hip[1]) ** 2) ** 0.5
        if torso < 1:
            continue
        r = sh / torso
        ratios.append(round(r, 2))
        votes["dtl" if r < 0.35 else "face_on" if r > 0.55 else "unknown"] += 1
    lm.close()
    if not ratios:
        return "unknown", "骨格を検出できず"
    # 正面なら少なくともアドレス付近で肩幅/胴長が 0.55 を超える。後方 (DTL) は始終 0.35 未満。
    mx = max(ratios)
    cam = "face_on" if mx > 0.55 else "dtl" if mx < 0.35 else "unknown"
    return cam, f"推定 (MediaPipe 肩幅/胴長 最大 {mx:.2f}, 値 {ratios})"


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
    estimate = "--estimate-camera" in args
    prev = None
    if "--previous" in args:
        pp = Path(args[args.index("--previous") + 1])
        if pp.exists():
            prev = json.loads(pp.read_text(encoding="utf-8"))
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
        if estimate and (cam == "unknown" or str(meta.get("camera_note", "")).startswith("推定")):
            cam, note = estimate_camera(lab, meta, n_samples=4)
            if cam != "unknown":
                meta["camera"] = cam
                meta["camera_note"] = note
                lab.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[camera] {meta['clip_id']}: {cam}  {note}")
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
    if prev is not None:
        prev_ids = {c["clip_id"] for c in prev.get("clips", [])}
        added = [c for c in clips if c["clip_id"] not in prev_ids]
        removed = prev_ids - {c["clip_id"] for c in clips}
        pt = prev.get("totals", {})
        print(f"=== 前回 ({Path(args[args.index('--previous') + 1]).name}) からの増分 ===")
        print(f"クリップ {len(prev_ids)} → {len(clips)} 本 (+{len(added)}" + (f", 削除 {len(removed)}" if removed else "") + ")")
        print("点の総数 " + f"{sum(pt.values())} → {sum(totals.values())} (+{sum(totals.values()) - sum(pt.values())})")
        print("由来の増分: " + ", ".join(f"{k} +{totals.get(k, 0) - pt.get(k, 0)}" for k in sorted(set(totals) | set(pt))))
        pc = prev.get("cameras", {})
        print("撮影方向の増分: " + ", ".join(f"{k} {pc.get(k, 0)} → {cams.get(k, 0)}" for k in sorted(set(cams) | set(pc))))
        print(f"強ラベル (holdout 除く): {prev.get('n_strong_usable', '?')} → {n_strong}")
        if added:
            print("新しいクリップ: " + ", ".join(f"{c['source']} ({c['camera']}, 強{c['strong']}/負{c['negative']})" for c in added))
        print()
    if problems:
        print("要確認:")
        for p_ in problems:
            print("  - " + p_)
    else:
        print("要確認: なし")
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        delta = None
        if prev is not None:
            prev_ids = {c["clip_id"] for c in prev.get("clips", [])}
            pt = prev.get("totals", {})
            delta = {"clips_added": [c["clip_id"] for c in clips if c["clip_id"] not in prev_ids],
                     "points": sum(totals.values()) - sum(pt.values()),
                     "sources": {k: totals.get(k, 0) - pt.get(k, 0) for k in sorted(set(totals) | set(pt))},
                     "cameras_prev": prev.get("cameras", {}), "n_strong_prev": prev.get("n_strong_usable")}
        out_json.write_text(json.dumps({"root": str(root), "clips": clips, "totals": dict(totals), "cameras": dict(cams),
                                        "sizes": dict(sizes), "problems": problems, "n_strong_usable": n_strong,
                                        "delta": delta},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
