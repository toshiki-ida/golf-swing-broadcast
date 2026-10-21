# CLAUDE.md — Golf Swing Broadcast System

プロ放送向けゴルフスイング軌道オーバーレイシステム。
DeckLink SDI 入出力・録画(グローウィング)・In/Out 編集・手描き軌道・SDI 送出・ShuttlePRO v2 操作を
1 本の CustomTkinter アプリ (`app.py`) に統合している。Windows 専用。

## 起動・ビルド

```powershell
# 開発環境 (リポジトリ直下に .venv を作る。.gitignore 未登録なので commit しないこと)
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py                      # 既定プロジェクト: ~/GolfSwingBroadcast
.\.venv\Scripts\python.exe app.py --project D:\golf    # プロジェクトフォルダ指定
.\.venv\Scripts\python.exe shuttle_diag.py             # ShuttlePRO HID バイト診断ツール

# exe (ローカル / onedir)
pip install pyinstaller
pyinstaller GolfSwingBroadcast.spec --noconfirm        # → dist/GolfSwingBroadcast/
```

- **ffmpeg 必須** (録画・切り出し・書き出し)。`ffmpeg_writer.find_ffmpeg()` が
  exe 同梱 → スクリプト同階層 → PATH → `C:\ffmpeg\bin` の順に探す。無ければ cv2 mp4v にフォールバック(低品質)。
- DeckLink は `C:/Program Files/Blackmagic Design/Blackmagic Desktop Video/DeckLinkAPI64.dll`
  を comtypes で読む。**無い PC では自動的にフォールバック(テストパターン)で起動する**
  (README は「Web カメラ」と書いているが、`DeckLinkInput._start_fallback` は
  `test_pattern_only=True` を渡しており Web カメラは開かない)。
- テストは存在しない。動作確認は実機 GUI か、`GolfBroadcastApp` を生成して
  `after()` で `_on_close()` を呼ぶスモークスクリプトで行う。

### ビルド経路が 2 つあり内容が異なる(注意)

| 経路 | 形式 | ffmpeg 同梱 | 備考 |
|------|------|-------------|------|
| `GolfSwingBroadcast.spec` | onedir, UPX | `C:\ffmpeg\bin` から同梱 | `collect_all('numpy'/'scipy')`, torch/tensorflow を excludes |
| `.github/workflows/build.yml` | **onefile** (`--onefile --windowed` + `--hidden-import` 列挙) | **なし** | spec を使わない。main への push で実行し `dist/*.exe` を Artifact 化 |

新しい依存(データファイル同梱が必要なもの)を足すときは **両方** を更新すること。

## ファイル構成と役割

| ファイル | 役割 |
|---------|------|
| `app.py` | GUI 全体 (`GolfBroadcastApp(ctk.CTk)`)。5 タブ分の UI 構築・イベント・スレッド橋渡し・設定永続化・ShuttlePRO/キーボード割当。約 5,000 行の単一クラス |
| `decklink_io.py` | DeckLink SDK (COM/ctypes) ラッパー。`DeckLinkInput`/`DeckLinkOutput` が自動フォールバック付き統合クラス。内部に `DeckLinkCaptureDevice`(入力)/`DeckLinkOutputDevice`(スケジュール出力)/`OpenCVCaptureDevice`(フォールバック)/`DummyOutputDevice` |
| `field_processor.py` | インターレース処理の Strategy。`NormalFrameProcessor`(bob デインターレース→59.94fps)と `HFRFieldProcessor`(Sony HFR 2x、フィールド独立展開)。`CaptureMode` enum と表示ラベル・公称 fps 表 |
| `recorder.py` | `Recorder`: FFmpegWriter への書き込み + グローウィング用リングバッファ(480x270 プレビュー + JPEG)。録画中の切り出し `export_clip`、停止後の高品質再書き出し `re_export_clip` |
| `ffmpeg_writer.py` | `FFmpegWriter`(rawvideo パイプ→libx264/NVENC/QSV/AMF)、`ffmpeg_extract`(フレーム範囲切り出し)、`find_ffmpeg`、`detect_hw_encoder` |
| `clip_manager.py` | `ClipData`(dataclass)、`TrajectoryData`、`ClipManager`(clips.json 永続化、In/Out、トリム書き出し、軌道 JSON の保存/読込) |
| `trajectory.py` | 軌道描画。`TimedSpline`(Catmull-Rom / ベジェ、フレーム同期)、`draw_gradient_trail`、`render_trajectory_on_frame` |
| `playout.py` | `PlayoutEngine`: プレイリスト、cue/play/pause/seek、先読みリーダースレッド、DeckLink スケジュール出力、playout.json 永続化、exports/ スキャン |
| `shuttle_pro.py` | ShuttlePRO v2 HID ドライバ(hidapi)。レポート形式自動検出、ジョグ/シャトル/ボタンをコールバック通知 |
| `ai/` | AI 試作。`club_tracker.py` (骨格基点のクラブヘッド追跡)、`swing_analyzer.py` (swingcut 由来の局面・指標・撮影方向判定・rules.yaml コメント)、`sns_render.py` (縦型動画、軌道由来と骨格由来の局面を統合)、`edit_hook.py` (編集タブの AI 操作。app.py からの唯一の入口。書き出し時に `dataset.py` でラベルを蓄積)、`config.py` (settings.json の "ai" 既定値)、`store.py` (`analysis/{clip_id}/`)、`club_detector.py` (ONNX Runtime CPU のヘッド検出器。`models/clubhead.onnx` が無ければ無効)、`dataset.py` (書き出し時に承認済みヘッド座標を `datasets/clubhead/<clip_id>/` に保存。holdout 除外) |
| `config/rules.yaml` | フィードバックコメントのルール (DTL は swingcut と同じ、`angle: face_on` が正面用)。exe では exe 同階層の config/ を読む |
| `tools/` | `label_club.py` (正解ラベル)、`eval_club.py` (誤差評価、`--all` で tests/real を一括、`--detector off|on|both` で検出器なし/ありを同じ表に)、`run_club_tracker.py` / `run_autotrace.py` (予測 JSON + 確認動画)、`import_labels.py` (tests/club_labels.json → datasets/clubhead)、`train_clubhead.py` (datasets/clubhead から検出器を学習し `models/clubhead.onnx` を書き出す。torch 入りの別 venv で実行) |
| `docs/AI_INTEGRATION_DESIGN.md` / `docs/CLUB_TRACKING_EVAL.md` | AI 統合設計と、autotrace (YOLO) vs club_tracker の比較評価 |
| `docs/HFR_double_speed_recording.md` | HFR 2x の設計メモ。**「通常モードは 2 枚中 1 枚を間引いて 29.97fps 録画」と書かれているが現行コードは `recording_frame_divisor=1`、通常モードは 59.94fps 全フレーム録画**。コードが正 |
| `NOTION_DOC.md` / `README.md` | 利用者向け概要。行数表などは古い |

## データの流れ

```
DeckLink SDI (1080i59.94)  ─ COM callback (memmove+enqueue のみ)
   └ DeckLinkCaptureDevice._process_loop  UYVY→BGR → field_processor (bob or HFR) → _deliver_frame
        ├ get_frame()  → app._start_capture_preview (after 33ms ループ, GUI)
        └ frame_callback → app._on_capture_frame → _capture_queue → _capture_write_loop → Recorder.write_frame
                                                                                    ├ FFmpegWriter (recordings/MM-DD/rec_*.mp4)
                                                                                    └ グローウィングバッファ (録画中スクラブ/切り出し)
録画停止 → _finish_rec_stop → ClipManager.add_clip (clips.json) → クリップタブ
クリップ (In/Out) → export_trimmed → clips/MM-DD/*_trim_*.mp4  (clip.exported_path)
編集タブ: FrameCache (JPEG キャッシュ) + 手描き制御点 → trajectories/{clip_id}_trajectory.json
  「動画書き出し」 _do_edit_export: 軌道焼き込み → yuv420p 一時 raw → ffmpeg(NVENC) → exports/MM-DD/swing_{name}.mp4
     → _add_export_to_playout (送出リストに自動追加)
送出タブ: PlayoutEngine → DeckLinkOutput.send_frame (BGRA, frame_duration で可変速) → SDI
```

### fps の意味 (間違えやすい)

| モード | DeckLink callback | 配信/プレビュー | 録画 fps (`recording_fps`) | 再生 |
|--------|------------------|----------------|---------------------------|------|
| 通常 | 29.97 (1 インターレースフレーム) | 59.94 (bob 2 枚) | **59.94** 全フレーム | 等倍 |
| HFR 2x | 29.97 | 59.94 (フィールド=別時刻) | **29.97** 全フレーム | 自動的に 1/2 速スロー (1 フレーム = 実時間 1/119.88 秒) |

録画ファイルは **音声トラックなし**(`FFmpegWriter` は映像のみ、COM コールバックの `audioPacket` は未使用、`EnableAudioInput` 未呼出)。

## プロジェクトフォルダ (既定 `~/GolfSwingBroadcast`)

```
recordings/MM-DD/rec_MMDD_HHMMSS.mp4   収録本体
clips/MM-DD/                           切り出し・トリム・取り込みファイル
exports/MM-DD/swing_*.mp4              軌道焼き込み済み送出用
trajectories/{clip_id}_trajectory.json 軌道 (swings + edit_in/edit_out)
clips.json / playout.json / settings.json
```

- `settings.json` の主なキー: `project_dir`, `record_dir`, `input_device`, `output_device`, `width/height/fps`,
  `capture_mode` ("normal"/"hfr_2x"), `crf`, `growing_buffer_sec`, `trajectory_style`, `shuttle_buttons`,
  `shuttle_pos_mapping`, `keyboard_shortcuts`, `*_sash_x`。`AppSettings` は既定 dict に `update()` するので
  新キーは `AppSettings.__init__` の既定に追加するだけでよい。
- `project_dir` は既定フォルダの settings.json に「ポインタ」として保存され、起動時にそこへ切り替わる。
- 学習用ラベルはプロジェクトフォルダではなく **リポジトリ直下の `datasets/clubhead/`** に貯まる (`ai/dataset.py` の `DATASET_DIR`)。
- クリップ/送出/編集の一覧は `source_path` の親ディレクトリ名 (`MM-DD`) で日付グループ化される。

## 設計上の約束ごと

- **GUI はメインスレッドのみ**。バックグラウンドから UI を触るときは必ず `self.after(0, ...)`。
  重い処理(書き出し・再書き出し・cue)は `threading.Thread(daemon=True)` で行う。
- **`frame_callback` は即返す**。DeckLink からのフレームはキューに積むだけにし、ディスク I/O は別スレッド。
  `_raw_queue`(30)や `_capture_queue`(600)が溢れるとフレームドロップになる。
- 既存機能に影響する状態は `self.deck_input` / `self.deck_output` / `self.recorder` / `self.playout` / `self.clip_manager`
  の 5 つ。`_save_settings` は `Recorder` を作り直すので録画中に呼ばない前提。
- キーボードショートカットは `bind_all("<Key>")` 1 本で、`tab_contexts`(収録/クリップ/編集/送出)に
  登録されたタブでしか発火しない。ShuttlePRO のボタンは **送出タブ表示中のみ** 動作、ジョグ/シャトルもタブ名で分岐。
  → 新しいタブを足しても、これらの dict に加えない限り既存操作には影響しない。
- 軌道の `points` のフレーム番号は **clip.in_frame 基準の相対値**(playout/編集書き出しとも `frame - in_frame` で描画)。
- `os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"` を cv2 import 前に設定している(mp4v マルチスレッドデコードのクラッシュ対策)。外さない。
- ログは `print` 中心。`playout` ロガーだけ DEBUG。
- 日本語 UI。フォントは Meiryo UI。CustomTkinter の dark テーマ。

## 追跡精度の評価ルール (club_tracker を触るとき)

- **調整前の数値を必ず先に記録する**: `python tools/eval_club.py --all --tag before` を実行し、
  `tests/real/results/<日時>_before_<commit>.md` をコミットしてから `ai/club_tracker.py` を変更する。
- 調整後は `--tag after` で別ファイルに残す。結果ファイルは上書き・削除しない(履歴が比較材料)。
- 既存素材で下がった項目があれば、良くなった項目と一緒に報告する。平均だけでなく最大誤差を見る。
- ラベル (`tests/club_labels.json`) は評価中に動かさない。直す場合は別コミットで理由を書く。
- 現在のラベルは Claude の目視 (sample / IMG_1095 の 12 点)。実収録素材のラベルはオーナーが付ける。

### 学習ベース検出器 (clubhead.onnx) の運用

- 学習データは **書き出し時に自動で貯まる** (`ai/dataset.py`): 送出用書き出し・縦型書き出しのたびに、そのクリップの制御点を
  manual / auto_confirmed (強ラベル)、スプライン補間点を interpolated (弱ラベル) として `datasets/clubhead/<clip_id>/` に保存する。
  同じクリップの再書き出しはフォルダごと上書き。`tests/real/split.yaml` の holdout は保存しない (評価が汚れる)。
- `datasets/` と `models/` は .gitignore 済み (公開リポジトリに映像・モデルを入れない)。
- 学習: `..\golf-autotrace\.venv\Scripts\python.exe tools/train_clubhead.py` (CUDA)。クリップ単位で train/val を分け、
  結果は `models/clubhead_train.json` に残る (件数・検証 px 誤差・PCK)。
- `ai/club_tracker.track_club(use_detector=None)` は `models/clubhead.onnx` があれば検出器の候補を足す。
  検出器は **候補の提案だけ** で、連続性・欠損・区間・信頼度の判定は従来どおり。モデルが無ければ従来方式のまま。
- 検出器を学習し直したら `tools/eval_club.py --all --split tune --tag det_N --detector both` で従来/検出器ありを同じ表に残す。

## ブランチ運用ルール

- **main への直接 push は禁止。** 作業は必ず `feature/*` ブランチで行い、Pull Request で提出する。
- ローカルには `.git/hooks/pre-push` を入れて main への push をブロックする(`tools/install_hooks.py` で設置)。
- `feature/*` への push は CI を起動しない(`.github/workflows/build.yml` は main への push と手動実行のみ)。
- Public リポジトリなので、push 前に動画・画像・モデル・フォント・個人のパスや名前が入っていないか `git ls-files` で確認する。

## 依存関係 (`requirements.txt`)

opencv-python / numpy / scipy / customtkinter / Pillow / comtypes / hidapi。
現状はバージョン固定なし(2026-10 時点で numpy 2.4 / opencv-python 5.0 / customtkinter 6.0 が入る)。
`opencv-python` と `opencv-contrib-python` は同じ `cv2` パッケージを上書きし合うので **同居させない**。

## ブランチ

- `main`: 現行。`feature/ai-analysis`(このブランチ)は main と同一地点から分岐。
- `origin/feature/autotrace-slowmo`: 2026-09 の AI 機能試作(YOLO セグメンテーションでクラブヘッド自動追尾、RIFE/minterpolate でスロー化)。
  torch/ultralytics を exe に同梱する方針でサイズが数 GB。`app.py` に +850 行、設定タブに「収録後の自動処理」を追加。
  `docs/AUTOTRACE_SPEC.md` に仕様あり。main には未マージ。

## 関連リポジトリ

- `../swingcut`: スマホ動画向けのスイング解析 & SNS 縦型動画(1080x1920/30fps)自動生成の試作。
  MediaPipe Pose → 局面検出(打球音 / ボール ROI / 手首位置の 3 手法でインパクト検出)→ 縦型レンダリング。
  GUI は PySide6(本リポジトリには持ち込まない)。統合設計は会話ログ / 今後 `docs/` に置く設計メモを参照。
