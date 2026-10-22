"""ai.store — 解析結果の置き場所 ({project}/analysis/{clip_id}/)。

clips.json / ClipData には何も足さず、クリップ ID ごとのフォルダに JSON を置く。

    club_track.json   ai.club_tracker の結果 (フレーム番号は clip.in_frame 基準の相対値)
                      points / conf / confident_points / landmarks (骨格 33 点, 元解像度 px) / body_h
"""
from __future__ import annotations

import json
from pathlib import Path


def analysis_dir(project_dir, clip_id: str) -> Path:
    d = Path(project_dir) / "analysis" / clip_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_club_track(dir_path, res, in_frame: int, source_path: str) -> Path:
    """ClubTrackResult を club_track.json に保存する。フレーム番号は in_frame を引いて相対値にする。"""
    def rel(f):
        return f - in_frame

    data = {
        "source_path": str(source_path),
        "in_frame": int(in_frame),
        "points": [[int(x), int(y), rel(f)] for x, y, f in res.points],
        "confident_points": [[int(x), int(y), rel(f)] for x, y, f in res.confident_points()],
        "conf": {str(rel(f)): round(float(c), 2) for f, c in res.conf.items()},
        "weak_ranges": [[rel(a), rel(b)] for a, b in res.weak_ranges()],
        "body_h": round(float(res.body_h), 1),
        "quality": getattr(res, "quality", {}),
        "reliable": bool(getattr(res, "reliable", True)),
        "real_fps": float(getattr(res, "real_fps", 30.0)),
        "seconds": round(float(res.seconds), 2),
        "landmarks": {str(rel(f)): [[round(float(x), 1), round(float(y), 1)] for x, y in lm]
                      for f, lm in getattr(res, "landmarks", {}).items()},
        # 手首中点と手元接続フラグ (区間の品質指標の再計算・閾値調整用)
        "wrist": {str(rel(f)): [round(float(x), 1), round(float(y), 1)] for f, (x, y) in getattr(res, "wrist", {}).items()},
        "attached": [rel(f) for f, v in getattr(res, "attached", {}).items() if v],
    }
    p = Path(dir_path) / "club_track.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def load_club_track(dir_path) -> dict | None:
    p = Path(dir_path) / "club_track.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def landmarks_from_track(track: dict | None) -> dict:
    """club_track.json の landmarks を {rel_frame: [(x, y), ...]} に戻す (無ければ空)。"""
    if not track or not track.get("landmarks"):
        return {}
    out = {}
    for k, lm in track["landmarks"].items():
        f = float(k)
        out[int(f) if f.is_integer() else f] = [(float(x), float(y)) for x, y in lm]
    return out


def load_clip_ai_settings(dir_path) -> dict:
    p = Path(dir_path) / "settings.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_clip_ai_settings(dir_path, **kw) -> None:
    d = load_clip_ai_settings(dir_path)
    d.update(kw)
    (Path(dir_path) / "settings.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def save_auto_snapshot(dir_path, segments: list, undrawn: list, all_segments: list | None = None,
                       segment_quality: list | None = None) -> None:
    """自動生成した制御点 (描いた区間) と未描画区間を残す。corrections の突き合わせに使う。
    all_segments: ゲートで止めた区間も含む全区間の制御点 (「全区間を描く」用)、segment_quality: 各区間の指標。"""
    import time as _t
    (Path(dir_path) / "auto_points.json").write_text(json.dumps({
        "created": _t.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": [[[int(x), int(y), f] for x, y, f in seg] for seg in segments],
        "undrawn": [[a, b] for a, b in undrawn],
        "all_segments": [[[int(x), int(y), f] for x, y, f in seg] for seg in (all_segments or [])],
        "segment_quality": segment_quality or [],
    }, ensure_ascii=False), encoding="utf-8")


def record_corrections(dir_path, swings) -> dict | None:
    """保存された軌道と auto_points.json を突き合わせ、人が足した・動かした・消した点を corrections.json に記録する。

    記録形式: {"frames": {"<frame>": {"x", "y", "source": "auto"|"manual"|"moved", "auto_xy": [x, y]|null}},
               "deleted": [[x, y, frame], ...], "updated": "..."}
    """
    d = Path(dir_path)
    ap = d / "auto_points.json"
    if not ap.exists():
        return None
    auto = json.loads(ap.read_text(encoding="utf-8"))
    auto_pts = {f: (x, y) for seg in auto.get("segments", []) for x, y, f in seg}
    cur = {}
    for sw in swings:
        for x, y, f in getattr(sw, "points", []):
            cur[f] = (int(x), int(y))
    # 「全区間を描く」で足した区間: 1 点でも残っていれば自動点として扱う (丸ごと消えたのは Undo か区間ごとの却下なので負例にしない)
    for seg in auto.get("extra_segments", []):
        if any(f in cur for _, _, f in seg):
            auto_pts.update({f: (x, y) for x, y, f in seg})
    frames = {}
    for f, (x, y) in cur.items():
        if f in auto_pts:
            ax, ay = auto_pts[f]
            src = "auto" if (abs(ax - x) <= 1 and abs(ay - y) <= 1) else "moved"
            frames[str(f)] = {"x": x, "y": y, "source": src, "auto_xy": [ax, ay]}
        else:
            frames[str(f)] = {"x": x, "y": y, "source": "manual", "auto_xy": None}
    deleted = [[ax, ay, f] for f, (ax, ay) in auto_pts.items() if f not in cur]
    import time as _t
    out = {"frames": frames, "deleted": deleted, "undrawn_at_generation": auto.get("undrawn", []),
           "updated": _t.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "corrections.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
