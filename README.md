# Golf Swing Broadcast System

プロ放送向けゴルフスイング軌道オーバーレイシステム。
DeckLink SDI入出力、グローウィング録画、軌道編集、送出、ShuttlePRO v2 操作を統合管理。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

## 機能

| タブ | 機能 |
|------|------|
| **収録** | DeckLink SDI入力プレビュー、REC/STOP、グローウィング（録画中スクラブ・切り出し） |
| **クリップ** | 素材一覧、In/Out設定、トリム書き出し、ファイル追加 |
| **編集** | フレーム単位プレビュー、スイング軌道描画（スプライン補間・グラデーション） |
| **送出** | プレイリスト再生、DeckLink SDI出力、可変速（1/8x〜1x）、CUE/フレーム送り |
| **設定** | デバイス選択、フォルダ管理、ShuttlePRO v2 ボタンマッピング |

## システム要件

- Python 3.10+
- Windows 10/11
- [Blackmagic Desktop Video](https://www.blackmagicdesign.com/support/) (DeckLinkドライバ)

### 対応ハードウェア

| デバイス | 用途 |
|---------|------|
| UltraStudio HD Mini | SDI入出力 (1080i59.94) |
| Contour ShuttlePRO v2 | ジョグ/シャトル/ボタン操作 (任意) |

> DeckLink非搭載環境ではWebカメラフォールバックで動作します。

## セットアップ

```bash
git clone https://github.com/toshiki-ida/golf-swing-broadcast.git
cd golf-swing-broadcast
pip install -r requirements.txt
python app.py
```

プロジェクトフォルダを指定して起動:

```bash
python app.py --project D:/golf_project
```

## 処理パイプライン

```
DeckLink SDI入力
  │
  ├─ [収録] → recordings/rec_日時.mp4
  │               │
  │     [切り出し] → clips/clip_日時.mp4
  │                      │
  │              [編集] → 軌道描画 → exports/swing_名前.mp4
  │                                       │
  │                              [送出] → DeckLink SDI出力
  │
  └─ [フォールバック] → Webカメラ入力
```

## データフォルダ

```
GolfSwingBroadcast/
├── recordings/     # 収録素材
├── clips/          # 切り出しクリップ
├── exports/        # 送出用（軌道焼き込み済み）
├── clips.json      # クリップメタデータ
├── playout.json    # 送出リスト
└── settings.json   # アプリ設定
```

## ShuttlePRO v2

ジョグ/シャトル/15ボタンに対応。設定タブでインタラクティブにマッピング可能。

| 操作 | 動作 |
|------|------|
| ジョグ回転 | フレーム送り/戻り (±1F) |
| シャトル CW | 可変速再生 (1/8x〜1x) |
| シャトル CCW | 逆再生 (1/8x〜5x) |
| ボタン | PLAY/PAUSE, STOP, CUE, PREV/NEXT, 速度切替 等 |

## キーボードショートカット（送出タブ）

| キー | アクション |
|------|-----------|
| `Space` | PLAY / PAUSE |
| `Enter` | PLAY |
| `Escape` | CUE (頭出し) |
| `D` / `→` | +1F |
| `A` / `←` | -1F |
| `W` / `S` | +5F / -5F |
| `N` / `P` | NEXT / PREV |
| `1`〜`4` | 速度切替 |

## ビルド

GitHub Actionsで `main` ブランチへのプッシュ時に自動ビルド。
手動実行も可 (Actions → workflow_dispatch)。

成果物は Actions → Artifacts からダウンロード。

## ファイル構成

| ファイル | 行数 | 内容 |
|---------|------|------|
| `app.py` | 2,316 | メインGUI (CustomTkinter) |
| `decklink_io.py` | 855 | DeckLink SDK ラッパー (COM/ctypes) |
| `playout.py` | 375 | 送出エンジン |
| `shuttle_pro.py` | 328 | ShuttlePRO v2 HIDドライバ |
| `clip_manager.py` | 260 | クリップ管理・メタデータ |
| `recorder.py` | 211 | 録画エンジン・グローウィングバッファ |
| `trajectory.py` | 182 | 軌道スプライン描画 |
