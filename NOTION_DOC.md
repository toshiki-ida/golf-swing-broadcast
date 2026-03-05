# Golf Swing Broadcast System

プロ放送向けゴルフスイング軌道オーバーレイシステム。DeckLink入出力、録画、In/Out編集、軌道描画、送出を統合管理する。

---

## システム概要

| 項目 | 内容 |
|------|------|
| 言語 | Python 3 |
| GUI | CustomTkinter |
| 映像I/O | Blackmagic DeckLink SDK (COM/ctypes) |
| 映像処理 | OpenCV |
| ハードウェア | UltraStudio HD Mini (SDI入出力) |
| コントローラ | Contour ShuttlePRO v2 (USB HID) |
| 解像度 | 1920×1080 / 1080i59.94 |
| コーデック | mp4v (H.264) |

---

## ページ構成 (5タブ)

### 1. 収録 (Capture)
- DeckLink SDI入力のリアルタイムプレビュー
- REC / STOP 制御
- **グローウィング対応**: 録画中にスクラブ・In/Out設定・クリップ切り出しが可能
- メモリバッファ: 最大1800フレーム（約60秒 @30fps）
- プレビュー用縮小バッファ (480×270) + フルレスJPEGバッファ (Q85)

### 2. クリップ (Clips)
- 録画素材の一覧管理
- In/Out ポイント設定
- トリム書き出し (MP4再エンコード)
- ファイル追加（外部MP4の取り込み）
- メタデータは `clips.json` で永続化

### 3. 編集 (Edit)
- クリップのフレーム単位プレビュー
- スイング軌道の描画エディタ
  - マウスで制御点をプロット → スプライン補間
  - グラデーションカラー（黄→赤、シアン→青 等）
  - 厚み・透明度調整
- 軌道データは `{clip_id}_trajectory.json` に保存
- 「動画書き出し → 送出」で軌道焼き込みMP4を生成

### 4. 送出 (Playout)
- プレイリスト管理（複数クリップの順序制御）
- DeckLink SDI出力への送出
- 速度制御: 1x / 1/2 / 1/4 / 1/8
- CUE（頭出し）/ PLAY / PAUSE / STOP / PREV / NEXT
- シークバー + フレーム送り (±1F, ±5F)
- **一時停止中もDeckLink出力** (フレーム送り時にモニター確認可能)
- クリップ末尾で停止（自動進行なし = 放送運用向け）

### 5. 設定 (Settings)
- プロジェクトフォルダ / 録画フォルダ設定
- 解像度・FPS設定
- DeckLink 入力/出力デバイス選択
- **ShuttlePRO v2 ボタン設定**
  - インタラクティブ図: 位置クリック → ボタン押下でマッピング
  - 15ボタン全てにアクション割り当て可能
  - 設定は `settings.json` に永続化

---

## ShuttlePRO v2 対応

### ハードウェア仕様
| 項目 | 内容 |
|------|------|
| デバイス | Contour ShuttlePRO v2 |
| 接続 | USB HID (VID=0x0B33, PID=0x0030) |
| ジョグダイヤル | 内側リング、フレーム送り |
| シャトルリング | 外側リング、可変速再生 (-7〜+7) |
| ボタン | 15個 (上段4 + 中段5 + 下部6) |

### 操作マッピング

#### ジョグダイヤル
- 回転 → フレーム送り/戻り（1フレーム単位）
- 再生中に触れると自動的に一時停止 → フレームステップ

#### シャトルリング
| 位置 | 動作 |
|------|------|
| 0 (中央) | 一時停止 |
| +1 | スロー再生 1/8x |
| +2 | スロー再生 1/4x |
| +3 | スロー再生 1/2x |
| +4〜+7 | 通常速 1x |
| -1 | 逆再生 1/8x (タイマー方式) |
| -2 | 逆再生 1/4x |
| -3 | 逆再生 1/2x |
| -4 | 逆再生 1x |
| -5〜-7 | 高速逆再生 2〜5x |

#### ボタン（デフォルト割り当て）
| BTN | アクション |
|-----|-----------|
| 1 | 前クリップ |
| 2 | CUE (頭出し) |
| 3 | PLAY / PAUSE |
| 4 | 次クリップ |
| 5 | 速度 1x |
| 6 | 速度 1/2 |
| 7 | 速度 1/4 |
| 8 | 速度 1/8 |
| 9 | STOP |

---

## ファイル構成

```
golf-swing-broadcast/
├── app.py              # メインGUI (2,316行)
├── recorder.py         # 録画エンジン + グローウィングバッファ (211行)
├── playout.py          # 送出エンジン (375行)
├── clip_manager.py     # クリップ管理・メタデータ (260行)
├── decklink_io.py      # DeckLink SDK ラッパー (855行)
├── trajectory.py       # 軌道スプライン描画 (182行)
├── shuttle_pro.py      # ShuttlePRO v2 HIDドライバ (328行)
└── shuttle_diag.py     # ShuttlePRO 診断ツール
```

合計: 約4,500行

---

## データフォルダ構成

```
GolfSwingBroadcast/          # プロジェクトルート
├── recordings/              # 収録素材 (rec_日時.mp4)
├── clips/                   # 切り出しクリップ
├── exports/                 # 送出用 (軌道焼き込み済み)
├── clips.json               # クリップメタデータ
├── playout.json             # 送出リスト
└── settings.json            # アプリ設定
```

---

## 処理パイプライン

```
DeckLink SDI入力
  │
  ├─ [収録] → rec_日時.mp4 + メモリバッファ (JPEG Q85)
  │               │
  │     [切り出し] → clips/clip_日時.mp4 (再エンコード)
  │                      │
  │              [編集] → 軌道描画 → exports/swing_名前.mp4
  │                                       │
  │                              [送出] → DeckLink SDI出力
  │                                       ↓
  │                                   モニター表示
  │
  └─ [フォールバック] → OpenCV Webカメラ (DeckLink非搭載時)
```

---

## キーボードショートカット (送出タブ)

| キー | アクション |
|------|-----------|
| Space | PLAY / PAUSE |
| Enter / F5 | PLAY |
| Escape | CUE (頭出し) |
| N / F8 | 次クリップ |
| P / F1 | 前クリップ |
| D / → | +1フレーム |
| A / ← | -1フレーム |
| W | +5フレーム |
| S | -5フレーム |
| 1〜4 | 速度切替 |

---

## 起動方法

```bash
python app.py
python app.py --project D:/golf_project
```

## 依存関係

- Python 3.10+
- customtkinter
- opencv-python
- numpy
- Pillow
- scipy
- hidapi (ShuttlePRO用)
- comtypes (DeckLink SDK用, Windows)
