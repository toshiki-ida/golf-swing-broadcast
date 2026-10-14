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
        "seconds": round(float(res.seconds), 2),
        "landmarks": {str(rel(f)): [[round(float(x), 1), round(float(y), 1)] for x, y in lm]
                      for f, lm in getattr(res, "landmarks", {}).items()},
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
