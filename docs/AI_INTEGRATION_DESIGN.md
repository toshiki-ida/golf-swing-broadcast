# AI 解析 & SNS 縦型動画 統合設計

対象ブランチ: `feature/ai-analysis`
ベース: golf-swing-broadcast (CustomTkinter / DeckLink / ShuttlePRO)
取り込み元: `../swingcut` (MediaPipe Pose → 局面・インパクト検出 → 1080x1920 縦型動画)

## 1. 目的

収録したスイング素材に対して、以下を自動で行う。

1. **AI 自動化**: 骨格推定 → 局面検出(アドレス / トップ / インパクト / フィニッシュ)→ 指標計測 → 自動 In/Out
2. **SNS 縦型動画の自動作成**: 通常速 → 骨格付きスロー → サマリーカード の 1080x1920 / 30fps / H.264 動画

既存の収録・編集・送出・ShuttlePRO 操作には手を入れない。

## 2. 制約(オーナー提示)

| 制約 | 本設計での扱い |
|---|---|
| GUI は CustomTkinter に合わせる(PySide6 不可) | UI は `ai/panel.py` の `AIPanel(ctk.CTkFrame)` のみ。swingcut の `gui.py` は移植しない |
| SDI 入出力・インターレース処理はベースを使う | AI 側は `decklink_io` が作った録画ファイルを読むだけ。`field_processor` の HFR は `time_scale` として解析に反映 |
| 既存機能を壊さない | キーボードは `tab_contexts`、ShuttlePRO は「送出」タブ名で分岐しており、新タブには届かない。録画中は AI ジョブを待機させ CPU 競合を避ける |
| 新機能は別モジュール、app.py は最小変更 | `ai/` パッケージ。app.py は 4 カ所・20 行前後 |
| mediapipe のビルド影響を評価 | §8。Phase 0 (`feature/build-mediapipe`) で実測 |

## 3. 決定事項

### 3.1 打球音は Phase 1 では使わない

ベースの録画は音声トラックを持たない(`FFmpegWriter` は映像のみ、`audioPacket` 未使用)。
録画への音声追加は収録安定性に関わるため、**オーナーと合意後に別 PR** とする。
Phase 1 のインパクト検出は **ボール ROI 差分 + 手首位置** の 2 手法。
59.94p 素材なら 1 フレーム ≈ 16.7 ms、HFR 2x 素材なら ≈ 8.3 ms の時間分解能が得られる。
SNS 動画の音声は BGM モードまたは無音(`render.yaml` の `audio_mode`)。

### 3.2 HFR 判定は fps から推定し、AI 設定で手動上書き可能にする

`ClipData` にはキャプチャモードの情報がない。録画 fps は通常 59.94 / HFR 29.97 で固定なので、
既定では `fps < 45 → HFR (time_scale=4.0)`, それ以外 `→ 通常 (time_scale=2.0)` と推定する
(`time_scale` = 「30fps 解析系列 1 フレームあたりのソースフレーム数」。実時間 fps = `fps × 実時間倍率`)。
AI 設定に「素材の種類: 自動 / 通常 59.94p / HFR 2x / 外部 30p」を持たせ、クリップ単位で上書きできるようにする。
**録画時に sidecar(`rec_*.json` にモード・実効 fps を書く)方式は、Recorder への変更を伴うためオーナー相談事項**とする。

### 3.3 カメラアングルは AI 設定で DTL / 正面 を選択する

swingcut の指標(前傾角、頭の移動、腰の突っ込み、テンポ)は DTL 前提。
正面(フェースオン)では **指標とコメントを出さず、局面検出と縦型書き出しのみ** 行う。
サマリーカードは正面ではテンポのみ、または省略する。

## 4. モジュール構成

```
ai/
  __init__.py        AI_AVAILABLE の判定 (mediapipe 不在でも import 可能)。重い import はここでしない
  pose_engine.py     録画ファイルを cv2 で読み、間引き・縮小して MediaPipe Pose (VIDEO モード) → pose.json
  swing_analyzer.py  swingcut/analyze.py の移植。fps / time_scale / camera_angle をパラメータ化。打球音は無効
  sns_render.py      swingcut/render.py の移植。FrameSampler で任意 fps 素材に対応。軌道焼き込み。1080x1920 出力
  feedback_rules.py  config/rules.yaml の評価 (swingcut と同じ書式)
  jobs.py            単一ワーカースレッド。進捗コールバック。録画中はジョブを待機
  store.py           {project}/analysis/{clip_id}/ の読み書きと一覧。clips.json / ClipData は変更しない
  panel.py           AIPanel: タブ UI (一覧・結果表示・手動実行) と設定セクション (build_settings / collect_settings)
config/rules.yaml, config/render.yaml      ルール・書き出し設定 (exe では exe 同階層に展開し編集可能)
assets/fonts/NotoSansJP.ttf                テロップ用フォント (setup_assets.py で取得)
models/pose_landmarker_heavy.task          MediaPipe モデル (setup_assets.py で取得)
```

swingcut の純粋関数(`extract_tracks`, `smooth`, `detect_impact`, `compute_metrics`, `compute_focus_rect`,
`draw_texts`, `make_summary_card`, `build_audio_track` 等)はコピーして最小修正で使う。
git submodule にはしない(PySide6 と `paths.py` の前提が異なるため)。

## 5. app.py への変更(最小)

1. `import` と、`_build_ui` で `tabview.add("AI/SNS")` を「設定」の前に追加し `self.ai_panel = AIPanel(tab, self)` を生成
2. `_finish_rec_stop` と `_finish_extract` の末尾で `self.ai_panel.on_clip_ready(clip.id)` を呼ぶ(自動処理の起点)
3. `_build_settings_tab` で `self.ai_panel.build_settings(tab)`、`_save_settings` で `self.ai_panel.collect_settings()`。
   `AppSettings` の既定 dict に `"ai": {...}` を 1 エントリ追加
4. `_on_tab_changed` に AI/SNS タブの分岐、`_on_close` に `self.ai_panel.shutdown()`

`AIPanel` は `app` への参照を受け取り、`app.clip_manager` / `app.settings` / `app.project_dir` /
`app.recorder.is_recording` / `app.after` だけを使う。app の private メソッドは呼ばない。

## 6. 処理の流れ

```
REC 停止 / クリップ切り出し
  └ (AI 設定: 自動実行 ON) → ai.jobs に投入 (録画中なら待機)
       1. pose_engine : cv2 読み → time_scale で間引き → (必要なら 960x540 に縮小) → Pose VIDEO → pose.json (+frame_map)
       2. swing_analyzer : 30fps 系列で局面検出 (ボール ROI → 手首位置) → 元フレーム番号へ戻す → analysis.json, phases/*.png
       3. 自動 In/Out (設定 ON): clip_manager.set_in_out(address - pre, finish + post)
       4. sns_render : FrameSampler で通常速 (30fps 実時間) / 1/4 スロー / 静止 / サマリー → analysis/{clip_id}/short.mp4
  └ AI/SNS タブ: 局面静止画・指標・コメント・縦型プレビュー・「出力フォルダを開く」・手動再実行
```

- `frame_map`: 解析系列のインデックス i → ソースフレーム `src = round(i × time_scale)`。局面は src で保存。
- 解析の窓幅(±4 フレーム、`find_peaks distance=5`、SavGol 窓)は 30fps 系列上で使うので swingcut の値のまま。
- 軌道(`trajectories/{clip_id}_trajectory.json`)があれば、クロップ前に `trajectory.render_trajectory_on_frame`
  で焼き込む。軌道のフレーム番号は `clip.in_frame` 基準の相対値。
- 出力は `{project}/analysis/{clip_id}/` にまとめ、`store.py` が一覧を返す。クリップ削除時の掃除は `AIPanel` が
  `clip_manager.clips` との差分で行う(app.py の削除処理には触れない)。

## 7. 設定(`settings.json["ai"]`)

| キー | 既定 | 内容 |
|---|---|---|
| `auto_run` | false | REC 停止 / 切り出し後に自動実行 |
| `auto_in_out` | false | 解析結果でクリップの In/Out を更新 |
| `camera_angle` | "dtl" | "dtl" / "face_on" |
| `source_kind` | "auto" | "auto" / "normal_5994" / "hfr_2x" / "external_30p" |
| `pose_model` | "heavy" | "heavy" / "lite"(速度優先) |
| `pose_downscale` | true | 960x540 に縮小して推論 |
| `run_during_rec` | false | 録画中もジョブを動かす |
| `audio_mode` | "mute" | "mute" / "bgm" |

## 8. ビルド・依存への影響

| 項目 | 内容 |
|---|---|
| mediapipe | 最新 1.1.0 は numpy 2 対応、jax 不要。`opencv-contrib-python` を要求する |
| cv2 の衝突 | `opencv-python` と `opencv-contrib-python` は同じ `cv2` を上書きし合う。requirements を contrib に置き換える |
| サイズ(onedir) | mediapipe 約 93 MB + モデル 30 MB + フォント 10 MB。matplotlib (15 MB) は `mp.solutions` を使わなければ excludes 可 |
| 起動 | mediapipe の import は約 1.4 秒。ワーカー内で遅延 import する |
| 速度 | heavy モデル CPU 1080p 19 ms/frame、540p 13 ms/frame(開発 PC)。GPU 不要 |
| spec | `collect_all('mediapipe')`、`models/` `assets/` `config/` の datas、`ai.*` hiddenimports、excludes に matplotlib |
| GitHub Actions | 現状は spec を使わず onefile で ffmpeg も未同梱。`setup_assets.py` → `pyinstaller GolfSwingBroadcast.spec` → zip に変更する(Phase 0 の実測後に別途) |

### 8.1 Phase 0 実測結果 (2026-10-14, `feature/build-mediapipe` d618099)

環境: Windows 11 / Python 3.11.9 / PyInstaller 6.22 / onedir / ffmpeg 未同梱(`C:\ffmpeg\bin` 不在のため両ビルドとも)。

| 項目 | main の spec | mediapipe 同梱 spec | 差 |
|---|---|---|---|
| 配布フォルダ | 319 MB | 467 MB | +148 MB |
| 内訳(増分) | - | mediapipe 61 / モデル 30 / matplotlib 15 / cv2 contrib 差分 +26 / フォント 10 | |
| 起動(cold, ウィンドウ出現まで) | 2.5 s | 3.0 s | +0.5 s |
| 起動(warm) | 1.2 s | 1.3 s | +0.1 s |

- **mediapipe 1.1.0** は numpy 2.4 / opencv-contrib-python 5.0 と共存し、jax を要求しない。
- **tasks API 互換**: `PoseLandmarker`(VIDEO モード)を venv と frozen exe の両方で実行し、swingcut(0.10.21)の
  `pose.json` と比較。検出率 100%、座標差は正規化座標で平均 0.0016 / 最大 0.0101(720p で 1〜2 px)。移植は無修正で可。
- **matplotlib は除外できない**: `mediapipe.tasks.python.vision.__init__` が `drawing_utils` 経由で無条件に import する。
  除外した frozen ビルドは `ModuleNotFoundError: matplotlib` で落ちることを確認。15 MB は受け入れる。
- **requirements.txt の日本語コメント**は pip が cp932 で読んで落ちるため、先頭に `# -*- coding: utf-8 -*-` を置いた。
- app.py は mediapipe を import しないため、起動時間の差は DLL/ファイル数の増加分のみ。
  Phase 1 でも mediapipe の import はワーカー内で遅延させる(import 自体に約 0.8〜1.7 s)。
- GitHub Actions は未変更。CI を spec ベース(onedir + zip)に切り替えるのは別コミット。

## 9. フェーズ

| Phase | 内容 | ブランチ |
|---|---|---|
| 0 | requirements の contrib 化、mediapipe tasks API 確認、spec 更新、ローカルビルドでサイズ・起動時間の実測 | `feature/build-mediapipe` |
| 1 | `ai/` 実装(pose / analyze / render / jobs / store / panel)、app.py 4 カ所、config / setup_assets | `feature/ai-analysis` |
| 2 | DeckLink 埋め込み音声の取得(sidecar WAV)→ 打球音インパクト検出・SNS 音声 | 別 PR(オーナー合意後) |
| 3 | 送出用 16:9 自動リプレイ(通常速 + スロー + 骨格)を exports/ に書き出し送出リストへ、正面用の指標 | 別 PR |

## 10. 関連ブランチ

`origin/feature/autotrace-slowmo`(2026-09)は YOLO セグメンテーションによるクラブヘッド自動追尾と
RIFE / minterpolate によるスロー化を app.py に直接組み込んでいる(+850 行、torch 同梱)。
本設計は別パッケージなので共存可能だが、両方を main に入れる場合は autotrace 側の app.py 変更を先にマージし、
その上に `ai/` の 4 カ所を載せる。重複・衝突の詳細は別途調査報告を参照。
