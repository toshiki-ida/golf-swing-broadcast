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

## AI 機能 (試作: クラブヘッド軌道の自動生成 / 縦型動画)

編集タブの「AI (試作)」に 2 つのボタンがある。

| ボタン | 内容 |
|------|------|
| ヘッド軌道を自動生成 | `ai/club_tracker.py` でクラブヘッドを追跡し、制御点を手描きと同じ軌道 (Swing) として保存。既存の点編集・ベジェ・Undo でそのまま手直しできる |
| 縦型動画 (SNS) 書き出し | 保存済みの軌道を主役に 1080x1920/30fps の動画を `exports/sns/` に書き出す (`ai/sns_render.py`) |

骨格線とサマリーカードは `settings.json` の `"ai": {"sns": {"skeleton": true, "summary": true}}` で ON/OFF (既定 ON)。
設定キーの一覧は `ai/config.py`。モデルとフォントは `python setup_assets.py` で取得する。

### 実収録素材での評価 (tests/real/)

実収録の 1080i59.94 素材が届いたら、次の手順で追跡精度を記録する。

```bash
# 1. 素材を置く (git 管理外)。1080i 素材はファイル名に "_i" を入れるとフィールド単位で処理される
cp D:/素材/rec_0611_130128.mp4 tests/real/rec_0611_130128_i.mp4

# 2. 主要フレームでクラブヘッドをクリックして正解ラベルを付ける (1 本 2〜3 分)
python tools/label_club.py tests/real/rec_0611_130128_i.mp4 --frames 100,130,160,175,185,200

# 3. 一括評価。結果は tests/real/results/<日時>_<tag>_<commit>.md に残る (上書きしない)
python tools/eval_club.py --all --tag before     # 調整前の数値を必ず先に記録する
#    ... ai/club_tracker.py を調整 ...
python tools/eval_club.py --all --tag after      # 調整後は別ファイルとして残す
```

確認動画は `build/club_eval/real/<素材名>/club_tracker_overlay.mp4`。
評価の詳細と autotrace との比較は `docs/CLUB_TRACKING_EVAL.md`。

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
