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
    # auto / dtl / face_on。auto はアドレス時の肩幅/胴長の比から推定 (斜めで判定不能なら指標を省く)
    "camera_angle": "auto",
    # 書き出し (送出用 / 縦型) のたびに、承認した軌道をクラブヘッド検出器の学習データとして
    # <プロジェクトフォルダ>/datasets/clubhead/ に保存する (ai/dataset.py)。False で保存しない
    "collect_labels": True,
    # クラブヘッド検出器 (models/clubhead.onnx, ONNX Runtime) をヘッド候補の提案に使うか。
    #   auto: モデルが同梱されていれば使う / off: 使わない (従来方式のみ)
    "club_detector": "auto",
    # 信頼度ゲート (区間ごとに判定。全部満たした区間だけ自動で描く)。
    # 優先順位: ここの既定 < config/ai.json (exe 同階層 → 同梱) < settings.json の "ai"。現場では config/ai.json を編集する。
    # 値の根拠: docs/CLUB_TRACKING_EVAL.md §10 (tune 11 本の区間ごとの目視判定、tests/real/segment_review.yaml)
    "gate": {
        "min_observed": 6,               # 区間の観測フレーム数
        "min_rate": 0.35,                # 観測率 = 観測フレーム / 区間の長さ
        "min_continuity": 0.70,          # 連続する観測の移動量が上限内だった割合
        "min_hand_connect_ratio": 0.0,   # 手元から伸びるシャフトに繋がっていた観測の割合 (正しい区間でも 0% が多いので使わない)
        "min_motion_bh": 0.3,            # 区間内でヘッドが動いた道のり (身長代替値比)。静止した物を追った区間を止める
        "max_on_head_ratio": 0.3,        # 選手の頭 (鼻・耳から 0.25 身長以内) に乗っていた観測の割合の上限。帽子追尾を止める
        "max_dist_std": 0.19,            # 手首との距離 (身長比) の標準偏差の上限。観客・背景を追うと大きい
        "max_dist_change": 0.07,         # 手首との距離の 1 フレーム変化 (身長比) の平均の上限
        "min_speed_ratio": 1.0,          # ヘッド速度 / 手首速度 の下限。ヘッドが手より遅いのは観客など
        "min_wrist_ratio": 0.05,         # 手の道のり / ヘッドの道のり の下限 (ヘッドが 2 身長以上動いた区間だけ判定)
    },
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


def config_file_path():
    """config/ai.json の場所。exe では exe 同階層の config/ (編集可能) を優先し、無ければ同梱 (_internal/config)。"""
    import sys
    from pathlib import Path
    if getattr(sys, "frozen", False):
        beside = Path(sys.executable).parent / "config" / "ai.json"
        if beside.exists():
            return beside
    from ai import resource_root
    return resource_root() / "config" / "ai.json"


def load_config_file() -> dict:
    """config/ai.json (無ければ空)。"_" で始まるキーはコメント扱いで無視する。"""
    import json
    p = config_file_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return _strip_comments(data) if isinstance(data, dict) else {}
    except Exception as e:  # pragma: no cover
        print(f"[AI] config/ai.json を読めません ({e})。既定値で動きます")
    return {}


def _strip_comments(d: dict) -> dict:
    return {k: (_strip_comments(v) if isinstance(v, dict) else v) for k, v in d.items() if not str(k).startswith("_")}


def get_ai_settings(settings) -> dict:
    """AppSettings (または dict) から "ai" 設定を取り出し、既定値 ← config/ai.json ← settings.json の順にマージして返す。"""
    data = getattr(settings, "data", settings) or {}
    user = data.get("ai", {}) if isinstance(data, dict) else {}
    merged = copy.deepcopy(AI_DEFAULTS)
    _deep_update(merged, load_config_file())
    _deep_update(merged, user)
    return merged


def gate_thresholds(settings=None) -> dict:
    """信頼度ゲートの閾値 (既定 ← config/ai.json ← settings.json)。"""
    return dict(get_ai_settings(settings or {})["gate"])


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
