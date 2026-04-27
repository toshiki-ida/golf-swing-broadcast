# Blackmagic DeckLink 倍速収録（HFR 2x）対応

## 概要

Blackmagic DeckLink（UltraStudio HD Mini）の **1080i59.94 SDI入力**を使い、Sony CCU の **HFR 2x（ハイフレームレート）** 信号を受け取って **1/2速スローモーション素材** を生成する機能。

> **通常収録との違い**: 同じ 1080i59.94 の SDI ケーブル1本で、実質 **119.88fps 相当** の映像を収録できる。ゴルフスイングのような高速モーションの分析・放送に最適。

---

## 仕組み

### Sony HFR 2x とは

Sony のカメラコントロールユニット（CCU）が提供するハイフレームレートモード。SDI 信号フォーマットは通常の **1080i59.94** のまま、Top Field / Bottom Field にそれぞれ **時間的に異なるフレーム** を格納して送出する。

```
通常のインターレース映像:
  Top Field    = 時刻T の偶数ライン
  Bottom Field = 時刻T の奇数ライン   ← 同じ瞬間の映像

Sony HFR 2x:
  Top Field    = 時刻T の全画素 (偶数ラインに格納)
  Bottom Field = 時刻T+Δ の全画素 (奇数ラインに格納)   ← 別の瞬間の映像
```

### フレーム処理フロー

```
DeckLink SDI入力 (1080i59.94)
│
├─── 通常モード ──────────────────────────────────────────┐
│    bob デインターレース                                  │
│    Top → 補間 → Frame A ─┐                             │
│    Bottom → 補間 → Frame B ─┤→ 59.94fps               │
│    録画: 29.97fps (1/2間引き) → 等倍再生               │
│                                                         │
├─── 倍速 HFR 2x モード ────────────────────────────────┐
│    フィールド独立展開                                    │
│    Top Field → upscale → Frame A (時刻T)   ─┐          │
│    Bottom Field → upscale → Frame B (時刻T+Δ) ─┤       │
│    → 119.88fps 相当                                     │
│    録画: 29.97fps (全フレーム記録) → 1/2速スロー再生   │
└─────────────────────────────────────────────────────────┘
```

---

## モード比較

| 項目 | 通常モード | 倍速 HFR 2x |
|------|-----------|-------------|
| **SDI入力** | 1080i59.94 | 1080i59.94（同一） |
| **フィールド処理** | bob デインターレース | フィールド独立展開 |
| **実効フレームレート** | 59.94 fps | 119.88 fps 相当 |
| **録画fps** | 29.97 fps（間引き） | 29.97 fps（全記録） |
| **再生速度** | 等倍（リアルタイム） | **1/2速スローモーション** |
| **フレーム間引き** | 2フレーム中1つ記録 | なし（全フレーム記録） |
| **ファイル尺** | 実時間と同じ | 実時間の **2倍** |
| **送出時** | 等倍再生 | 自動スロー再生 |

---

## 技術詳細

### フィールド分離 (`field_processor.py`)

Strategy パターンで通常モード / HFR モードを差し替え可能にしている。

```python
def extract_top_field(frame: np.ndarray) -> np.ndarray:
    """偶数ライン (0, 2, 4, ...) を抽出 → 1920x540"""
    return frame[0::2]

def extract_bottom_field(frame: np.ndarray) -> np.ndarray:
    """奇数ライン (1, 3, 5, ...) を抽出 → 1920x540"""
    return frame[1::2]

def upscale_field_to_frame(field: np.ndarray) -> np.ndarray:
    """540ライン → 1080ライン に線形補間で拡張"""
    h, w = field.shape[:2]
    return cv2.resize(field, (w, h * 2), interpolation=cv2.INTER_LINEAR)

def build_high_frame_rate_frames(frame, upper_field_first=True):
    """1入力インターレースフレーム → 2つの独立プログレッシブフレーム"""
    top    = upscale_field_to_frame(extract_top_field(frame))
    bottom = upscale_field_to_frame(extract_bottom_field(frame))
    return [top, bottom] if upper_field_first else [bottom, top]
```

HFR プロセッサは `emit()` コールバックで1入力→2出力を配信する:

```python
class HFRFieldProcessor(IFrameProcessor):
    """倍速モード: フィールドを「別フレーム」として独立展開"""

    def process(self, bgr, upper_field_first, emit, input_frame_no=0):
        frames = build_high_frame_rate_frames(bgr, upper_field_first)
        for frame in frames:
            emit(frame)       # → プレビュー表示 + 録画キュー
            self._output_frame_no += 1
```

| クラス | モード | 処理内容 |
|--------|--------|----------|
| `NormalFrameProcessor` | 通常 | bob デインターレース（フィールド間補間） |
| `HFRFieldProcessor` | 倍速 | フィールド独立展開（補間なし） |

### DeckLink コールバックからの呼び出し (`decklink_io.py`)

DeckLink COM コールバックで受け取った UYVY 生データを BGR に変換し、モードに応じたプロセッサに渡す:

```python
def _on_frame_arrived(self, frame_data_copy, width, height, row_bytes):
    # UYVY → BGR 変換
    uyvy = np.frombuffer(frame_data_copy, dtype=np.uint8).reshape(height, row_bytes)
    uyvy_img = uyvy[:, :width * 2].reshape(height, width, 2)
    bgr = cv2.cvtColor(uyvy_img, cv2.COLOR_YUV2BGR_UYVY)

    if height >= 720 and self._interlaced:
        # Strategy パターン: 通常 or HFR のプロセッサが自動選択される
        self._processor.process(
            bgr, self._upper_field_first,
            self._deliver_frame,       # 出力先コールバック
            self._input_frame_no,
        )
    else:
        self._deliver_frame(bgr)       # プログレッシブ: そのまま
```

### 録画時のフレーム間引き (`app.py`)

```python
def _on_capture_frame(self, frame):
    """通常: bob 2倍出力の半分を間引き / HFR: 全フレーム記録"""
    if self.recorder.is_recording:
        divisor = self.deck_input.recording_frame_divisor  # 通常=2, HFR=1
        if divisor > 1:
            self._rec_frame_count += 1
            if self._rec_frame_count % divisor != 0:
                return              # 間引き (通常モードのみ)
        self._capture_queue.put_nowait(frame)
```

```
recording_frame_divisor:
  通常モード → 2  (bob 2倍出力の半分を間引き → 等倍再生)
  HFR モード → 1  (全フレーム記録 → スローモーション)

recording_fps:
  両モードとも 29.97fps (DeckLinkコールバック実測値)
```

### モード動的切替 (`decklink_io.py`)

収録中でもプロセッサだけを差し替え、コールバックチェーンを止めずに切り替え可能:

```python
@capture_mode.setter
def capture_mode(self, mode: CaptureMode):
    """モードを動的に切り替える (キャプチャ中でも可)"""
    self._capture_mode = mode
    self._processor = make_processor(mode)   # Strategy 差し替え

def make_processor(mode: CaptureMode) -> IFrameProcessor:
    if mode == CaptureMode.HighFrameRate2x:
        return HFRFieldProcessor()
    return NormalFrameProcessor()
```

### 送出時の自動スロー再生 (`playout.py`)

DeckLink スケジュール出力の `frame_duration` がクリップの fps に基づくため、29.97fps で記録された HFR クリップは自動的に **1/2速スローモーション** で再生される。特別な速度制御は不要。

---

## UI操作

### モード切替

収録タブの **「入力モード」** セグメントボタンで切り替え。

| ボタン表示 | 値 | 説明 |
|-----------|-----|------|
| 通常 (59.94fps) | `normal` | 標準的なデインターレース収録 |
| 倍速 HFR 2x (29.97fps記録 → 1/2速スロー) | `hfr_2x` | ハイフレームレート倍速収録 |

- **収録中でもモード切替可能**（DeckLink に即時反映）
- 実効 fps がモードラベル横にリアルタイム表示
- 設定は `settings.json` に自動保存（次回起動時に復元）

### 録画から送出までの流れ

```
1. 収録タブで「倍速 HFR 2x」を選択
2. REC → STOP で録画  (29.97fps で記録される)
3. クリップタブで In/Out 設定・トリム
4. 編集タブで軌道描画
5. 送出タブに追加 → PLAY
   → 自動的に 1/2速スローモーションで SDI 出力
```

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                    DeckLink SDI 入力                  │
│                    (1080i59.94)                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  DeckLinkInput  (decklink_io.py)                     │
│  ┌────────────────────────────────────────────────┐  │
│  │ COMコールバック → raw_queue → _process_loop   │  │
│  │   UYVY → BGR → フィールド処理                  │  │
│  └─────────────────────┬──────────────────────────┘  │
│                        │                              │
│    ┌───────────────────┴───────────────────┐         │
│    ▼                                       ▼         │
│  NormalFrameProcessor              HFRFieldProcessor │
│  (bob deinterlace)              (フィールド独立展開) │
│  Top+Bottom → 補間 → 2F        Top → upscale → F_A  │
│                                 Bot → upscale → F_B  │
│    │                                       │         │
│    └───────────────────┬───────────────────┘         │
│                        ▼                              │
│              _deliver_frame()                         │
│              (プレビュー + コールバック)               │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    ┌──────────┐            ┌────────────┐
    │ プレビュー │            │   録画     │
    │ (GUI表示) │            │ Recorder   │
    └──────────┘            │            │
                            │ 通常: 間引き│
                            │ HFR: 全記録 │
                            └──────┬─────┘
                                   ▼
                            rec_日時.mp4
                            (29.97fps)
```

---

## 設計のポイント

### なぜ SDI 信号そのままで倍速が可能か

Sony CCU の HFR 2x は、既存の **1080i59.94 インフラ**（ケーブル・ルーター・フレームシンクロナイザー）をそのまま流用できる点が最大のメリット。DeckLink 側は通常の 1080i59.94 として受け取り、ソフトウェア側でフィールドの解釈を切り替えるだけで対応できる。

### Strategy パターンの採用理由

`NormalFrameProcessor` と `HFRFieldProcessor` を同一インターフェース（`IFrameProcessor`）で実装することで、**収録中でもモードを動的に切替可能**にしている。DeckLink のコールバックチェーンを止めることなく、プロセッサだけを差し替える設計。

### 録画 fps を 29.97 に統一する理由

HFR 2x では実効 119.88fps 相当のフレームが生成されるが、録画は **29.97fps** で行う。これにより：

- ファイルサイズが抑えられる（59.94fps 録画の半分）
- 再生時に自動的に **1/2速スロー** になる（fps メタデータに依存）
- 通常の動画プレーヤー（VLC 等）でもスロー再生として正しく表示される

---

## 補足: 将来の拡張ポイント

| 項目 | 現状 | 将来案 |
|------|------|--------|
| フィールドアップスケール | `cv2.INTER_LINEAR` | NNEDI3 / Lanczos / RIFE 等 |
| HFR倍率 | 2x 固定 | 3x / 4x 対応 |
| フィールドドミナンス | 上フィールド先行固定 | SDI信号から自動検出 |
| モーション補間 | なし | RIFE による中間フレーム生成 |
