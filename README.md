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

## 編集タブの操作 (軌道の点)

| 操作 | キー / ボタン | 動き |
|------|--------------|------|
| 点を置く | 左クリック | 現在フレームに制御点を置く |
| 点を動かす / ハンドル | 右ドラッグ / 右クリック | 点の移動、ベジェハンドルの ON/OFF |
| **現在フレームの点を削除** | `Delete` / 「この点を削除」 | 表示中のフレームに置かれた点を消す。`A`/`D` や ShuttlePRO のジョグで送りながら連続して消せる。点が無いフレームでは最寄りの点のフレームを表示 |
| **範囲削除** | `Shift+Delete` 2 回 / 「範囲削除 開始」→「ここまで」 | 1 回目で開始フレームを記憶、終点のフレームでもう 1 回押すと区間内の点をまとめて消す。`Escape` で取消。IN/OUT とは別 |
| 元に戻す / やり直す | `Ctrl+Z` / `Ctrl+Shift+Z` | 削除・追加・移動を戻す (削除で分かれたスイングの本数も戻る) |

- 消した点の前後は **補間しない**。消した点を挟んで軌道は 2 本のスイングに分かれ、消した区間は描画されない
  (再び描きたければ、その区間に点を置いて「区間を結合」)。
- タイムライン (IN/OUT バーの下段) にスイングごとの色で点の位置が出る。現在フレームは白い縦線、現在フレームに点があれば
  白い四角で強調し、フレーム番号の横に `●` (点あり) / `○` (点なし) が出る。画面上でも現在フレームの点は白い二重丸で囲む。
- 信頼度ゲートで止まった区間や観測の無い区間は、手首の軌跡と過去の承認済み軌道から **推定して埋める** (既定)。推定点は
  **中抜きの丸** (観測点・手描き点は塗りつぶし) で表示され、AI 欄に「推定で補った区間: F…-…」が出る。Delete / 移動は観測点と同じ操作。
  推定のまま書き出した点は学習データに入らない (動かして直した点だけ入る)。送出・縦型動画では観測と推定を同じ見た目で描く。
  推定を使わず従来どおり描かない設定: settings.json か config/ai.json の `"trajectory_fill": "off"`
- 自動生成の結果は信頼度ゲートで区間ごとに判定され、AI 欄に「ゲート: …」の閾値と「区間1 F139-179 描画 | 率83% …」「区間3 … 未描画 | … ← 手首距離ばらつき 0.23 > 0.19」
  のように各区間の値と止まった理由が出る。止まった区間も描きたいときは「全区間を描く (ゲート無視)」(Undo 可)。
- 自動生成された点を人が削除すると、書き出し時にその座標が **誤検出 (negative)** として学習データ
  (`<プロジェクト>/datasets/clubhead/`) に記録され、次回の学習で負例として使われる。削除キーは設定タブで変更できる。

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
設定キーの一覧は `ai/config.py`。軌道の信頼度ゲートの閾値は `config/ai.json` (exe 同階層に置けば同梱より優先) か
settings.json の `"ai": {"gate": {...}}` で変えられる (根拠: docs/CLUB_TRACKING_EVAL.md §10)。モデルとフォントは `python setup_assets.py` で取得する。

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
