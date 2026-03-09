"""
フィールド処理モジュール

通常モードと倍速モード（Sony HFR 2x）のフレーム処理戦略を定義する。

Sony HFR 2x モードについて:
  DeckLink入力信号: 1080i59.94
  ただし入力ソースは Sony CCU の HFR 2x を想定。
  1入力フレーム = Top Field / Bottom Field がそれぞれ時間的に異なるフレームに対応する。
  通常インターレース映像のように Top/Bottom を合成しない。
  各フィールドを別時刻フレームとして扱う。
  結果として、実効的に約119.88fps のフレーム列を得る。
"""

import enum
import logging
from typing import Callable, List

import cv2
import numpy as np

log = logging.getLogger("field_processor")


class CaptureMode(enum.Enum):
    Normal = "normal"            # 通常モード: bob デインターレース (59.94fps相当)
    HighFrameRate2x = "hfr_2x"  # 倍速モード: フィールド独立展開 (119.88fps相当)


# UI表示用ラベル
CAPTURE_MODE_LABELS = {
    CaptureMode.Normal:          "通常 (59.94fps)",
    CaptureMode.HighFrameRate2x: "倍速 HFR 2x (119.88fps)",
}

# 実効出力fps (録画・表示用)
CAPTURE_MODE_EFFECTIVE_FPS = {
    CaptureMode.Normal:          59.94,
    CaptureMode.HighFrameRate2x: 119.88,
}


# =============================================================================
# フィールド分離・復元ヘルパー (単体テスト可能な形で分離)
# =============================================================================

def extract_top_field(frame: np.ndarray) -> np.ndarray:
    """
    偶数ライン (0, 2, 4, ..., 1078) を抽出して 1920x540 フィールドを返す。
    Sony HFR 2x モード: このフィールドが時刻 T のフレームに相当する。
    """
    return frame[0::2]  # shape: (540, 1920, 3)


def extract_bottom_field(frame: np.ndarray) -> np.ndarray:
    """
    奇数ライン (1, 3, 5, ..., 1079) を抽出して 1920x540 フィールドを返す。
    Sony HFR 2x モード: このフィールドが時刻 T + 1/119.88s のフレームに相当する。
    """
    return frame[1::2]  # shape: (540, 1920, 3)


def upscale_field_to_frame(field: np.ndarray) -> np.ndarray:
    """
    540ライン → 1080ライン に線形補間で拡張する。

    将来的に高画質補間 (NNEDI3, RIFE, Lanczos 等) に置き換えやすいよう
    関数として独立させている。現在は速度優先で cv2.INTER_LINEAR を使用。

    Args:
        field: shape (H, W, 3) の numpy 配列 (例: 540x1920)
    Returns:
        shape (H*2, W, 3) の numpy 配列 (例: 1080x1920)
    """
    h, w = field.shape[:2]
    return cv2.resize(field, (w, h * 2), interpolation=cv2.INTER_LINEAR)


def build_high_frame_rate_frames(
    frame: np.ndarray, upper_field_first: bool = True
) -> List[np.ndarray]:
    """
    Sony HFR 2x: 1入力インターレースフレームから2つの独立フレームを生成する。

        Input Frame (1920x1080 interlaced)
          → Top Field    (even lines, time T)      → upscale → 1920x1080 frame A
          → Bottom Field (odd lines,  time T+1/119.88s) → upscale → 1920x1080 frame B

    Args:
        frame:             1920x1080 BGR インターレースフレーム
        upper_field_first: True=上フィールド先行 (通常の1080i)
    Returns:
        [earlier_frame, later_frame] の時系列順に2枚
    """
    top_frame    = upscale_field_to_frame(extract_top_field(frame))
    bottom_frame = upscale_field_to_frame(extract_bottom_field(frame))
    return [top_frame, bottom_frame] if upper_field_first else [bottom_frame, top_frame]


# =============================================================================
# フレーム処理 Strategy
# =============================================================================

class IFrameProcessor:
    """フレーム処理インターフェース (Strategy パターン)。

    通常モードと倍速モードを同一インターフェースで差し替え可能にする。
    """

    def process(
        self,
        bgr: np.ndarray,
        upper_field_first: bool,
        emit: Callable[[np.ndarray], None],
        input_frame_no: int = 0,
    ) -> None:
        raise NotImplementedError


class NormalFrameProcessor(IFrameProcessor):
    """
    通常モード: bob デインターレース。
    1入力フレーム → 2出力フレーム (59.94fps相当)。
    各フィールドの欠けラインを隣接フィールドの補間で埋める。
    """

    def process(
        self,
        bgr: np.ndarray,
        upper_field_first: bool,
        emit: Callable[[np.ndarray], None],
        input_frame_no: int = 0,
    ) -> None:
        h, w = bgr.shape[:2]
        if h >= 720:
            # 上フィールド: 偶数ラインはそのまま、奇数ラインは隣接偶数ラインで補間
            frame_upper = np.empty_like(bgr)
            frame_upper[0::2] = bgr[0::2]
            frame_upper[1:-1:2] = (
                (bgr[0:-2:2].astype(np.uint16) + bgr[2::2].astype(np.uint16)) >> 1
            ).astype(np.uint8)
            frame_upper[-1] = bgr[-2]

            # 下フィールド: 奇数ラインはそのまま、偶数ラインは隣接奇数ラインで補間
            frame_lower = np.empty_like(bgr)
            frame_lower[1::2] = bgr[1::2]
            frame_lower[0] = bgr[1]
            frame_lower[2::2] = (
                (bgr[1:-2:2].astype(np.uint16) + bgr[3::2].astype(np.uint16)) >> 1
            ).astype(np.uint8)

            if upper_field_first:
                emit(frame_upper)
                emit(frame_lower)
            else:
                emit(frame_lower)
                emit(frame_upper)
        else:
            emit(bgr)


class HFRFieldProcessor(IFrameProcessor):
    """
    倍速モード (Sony HFR 2x): フィールド独立展開。
    1入力フレーム → 2出力フレーム (119.88fps相当)。

    通常ボブデインターレースとの違い:
      各フィールドを「同一フレームの補間情報」ではなく、
      「時間的に独立した別フレーム」として扱う。
      フィールド間の合成・補間は行わない。
    """

    def __init__(self):
        self._output_frame_no = 0
        self._drop_count = 0

    def process(
        self,
        bgr: np.ndarray,
        upper_field_first: bool,
        emit: Callable[[np.ndarray], None],
        input_frame_no: int = 0,
    ) -> None:
        frames = build_high_frame_rate_frames(bgr, upper_field_first)
        field_names = ("Top", "Bottom") if upper_field_first else ("Bottom", "Top")
        for i, frame in enumerate(frames):
            log.debug(
                "[HFR] input#%d → %s field → output#%d",
                input_frame_no, field_names[i], self._output_frame_no,
            )
            emit(frame)
            self._output_frame_no += 1


def make_processor(mode: CaptureMode) -> IFrameProcessor:
    """モードに応じたプロセッサを生成する"""
    if mode == CaptureMode.HighFrameRate2x:
        return HFRFieldProcessor()
    return NormalFrameProcessor()
