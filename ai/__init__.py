"""ai — AI 解析パッケージ (骨格推定・クラブヘッド追跡・局面検出)。

重い依存 (mediapipe) はこのモジュールでは import しない。
各サブモジュールが必要になった時点で遅延 import する。

    from ai import ai_available
    if ai_available():
        from ai.club_tracker import track_club
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ai_available() -> bool:
    """mediapipe が import 可能か (実際には import しない)。"""
    return importlib.util.find_spec("mediapipe") is not None


def resource_root() -> Path:
    """同梱リソース (models/, assets/, config/) の基準ディレクトリ。

    PyInstaller 実行時は展開先 (sys._MEIPASS = <exe>/_internal)、開発時はリポジトリルート。
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def default_pose_model() -> Path:
    return resource_root() / "models" / "pose_landmarker_heavy.task"
