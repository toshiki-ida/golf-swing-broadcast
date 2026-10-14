"""ai.config — AI 機能の設定 (settings.json の "ai" キー) と既定値。

app.py の AppSettings には触らず、ここで既定値を持ち、settings.json に "ai" があれば上書きする。
設定タブの UI は未実装なので、変更は settings.json を直接編集する。
"""
from __future__ import annotations

import copy

AI_DEFAULTS = {
    # 素材の種類: auto / normal_5994 / hfr_2x / external_30p
    #   auto: fps が 29.97 (29.9〜29.99) なら HFR 2x 録画とみなす (ベースの録画 fps は 59.94 か 29.97 の 2 択)。
    #         30.00 など他の値は実時間 1 倍の素材として扱う。
    "source_kind": "auto",
    # dtl / face_on  (face_on では骨格指標を出さない)
    "camera_angle": "dtl",
    # 縦型 (SNS) 書き出し
    "sns": {
        "skeleton": True,       # スロー区間に骨格線を描く (club_track.json に骨格があるときのみ)
        "summary": True,        # 末尾にサマリーカードを付ける
        "slow_factor": 4,       # スロー区間の倍率 (実時間の 1/N)
        "freeze_sec": 0.7,      # トップ / インパクトの静止秒数
        "summary_sec": 3.0,     # サマリーカードの秒数
        "pre_frames": 6,        # アドレスの何フレーム前から入れるか (30fps 換算)
        "audio_mode": "mute",   # mute / bgm / source
        "bgm_path": "",         # audio_mode=bgm のときの BGM ファイル
        "layout": "auto",       # auto / crop / pad  (横位置素材を縦にする方法)
        "crf": 18,
        "hw_encode": True,
    },
}


def get_ai_settings(settings) -> dict:
    """AppSettings (または dict) から "ai" 設定を取り出し、既定値とマージした新しい dict を返す。"""
    data = getattr(settings, "data", settings) or {}
    user = data.get("ai", {}) if isinstance(data, dict) else {}
    merged = copy.deepcopy(AI_DEFAULTS)
    _deep_update(merged, user)
    return merged


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def realtime_factor(fps: float, source_kind: str = "auto") -> float:
    """素材 1 フレームが実時間の何フレーム分か (実時間 fps = fps × この値)。

    normal_5994 / external_30p → 1.0、hfr_2x → 4.0 (29.97fps 記録で 119.88fps 相当)。
    auto は fps が 29.97 のときだけ HFR とみなす。
    """
    if source_kind == "hfr_2x":
        return 4.0
    if source_kind in ("normal_5994", "external_30p"):
        return 1.0
    return 4.0 if 29.9 <= fps <= 29.99 else 1.0
