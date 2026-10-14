# クラブヘッド軌道の自動検出 — autotrace (YOLO) と club_tracker (骨格) の比較評価

日付: 2026-10-14 / ブランチ: `feature/ai-analysis`
評価環境: Windows 11, Core CPU + RTX 3050 Laptop (4 GB), Python 3.11

## 1. 対象

| 手法 | 実装 | 依存 |
|---|---|---|
| **autotrace** | `origin/feature/autotrace-slowmo` の `autotrace.py`(git worktree `../golf-autotrace` + 専用 venv で実行) | torch (CUDA 12.8) + ultralytics YOLO11m-seg + imageio-ffmpeg |
| **club_tracker** | `ai/club_tracker.py`(本ブランチ、試作) | mediapipe + opencv + numpy + scipy |

評価ツール: `tools/label_club.py`(正解ラベル作成)、`tools/eval_club.py`(誤差一覧)、
`tools/run_autotrace.py` / `tools/run_club_tracker.py`(予測 JSON + 確認動画の出力)。
正解ラベルは `tests/club_labels.json`(6 局面 × 2 本 = 12 点)。
**ラベルは Claude が 3 倍拡大画像で目視して付けたもの(±3 px 程度、IMG_1095 のインパクトは残像のみで ±15 px)。オーナーの確認が必要。**

## 2. 素材

| 素材 | 内容 | 備考 |
|---|---|---|
| `sample_30fps.mp4` | 720x1280 / 30p / 188 f。屋外・逆光・ドライバー・DTL(swingcut の sample.mov) | レンズフレア、長い影あり |
| `IMG_1095_1080p.mp4` | 1080x1920 / 30p / 269 f。練習場・アイアン・DTL(`Documents/swingcut/IMG_1095` の 4K を縮小) | 細いシャフト、背景に人工物 |
| `IMG_1095_interlaced.mp4` | 上を 2 フレームずつ織り込んだ合成 1080i 相当(15 fps, 134 f) | **1080/59.94i の実素材はこの PC に無い**ため、フィールド処理の動作確認用 |

## 3. 結果(正解との誤差 px、画面高さ比)

### sample_30fps.mp4 (720x1280)

| frame | 局面 | autotrace (In/Out 指定) | club_tracker |
|---|---|---|---|
| 14 | アドレス | 推定なし | 14.3 (1.1%) |
| 33 | テイクバック | 14.3 (1.1%) | 6.7 (0.5%) |
| 52 | トップ | 1848 (144%、画面外へ外挿) | 9.2 (0.7%) |
| 58 | ダウン | 推定なし | 1.4 (0.1%) |
| 63 | インパクト | 推定なし | 6.4 (0.5%) |
| 70 | フォロー | 推定なし | 1.0 (0.1%) |
| | **平均 / 最大** | 評価可能 2 点のみ | **6.5 / 14.3** |
| | 観測できたフレーム | 25 / 188(確信 9) | 108 / 188(確信) |
| | 処理時間 | 6.1 s (GPU) | 32.1 s (CPU、うち骨格 15.0 s) |

### IMG_1095_1080p.mp4 (1080x1920)

| frame | 局面 | autotrace (In/Out 指定) | club_tracker |
|---|---|---|---|
| 37 | アドレス | 推定なし | 14.8 (0.8%) |
| 52 | テイクバック | 推定なし | 11.0 (0.6%) |
| 66 | トップ | 推定なし | 14.2 (0.7%) |
| 70 | ダウン | 推定なし | 7.1 (0.4%) |
| 74 | インパクト | 推定なし | 5.0 (0.3%) |
| 86 | フォロー | 推定なし | 11.4 (0.6%) |
| | **平均 / 最大** | 評価不能(スイング区間に点なし) | **10.6 / 14.8** |
| | 観測できたフレーム | 8 / 66 | 116 / 269(確信) |
| | 処理時間 | 7.8 s (GPU) | 49.3 s (CPU、うち骨格 18.5 s) |

### 合成インターレース(club_tracker `--interlaced`、フィールド単位 = 119.88 fps 相当の処理)

| 局面 | 誤差 px |
|---|---|
| アドレス / テイクバック / トップ / ダウン / インパクト / フォロー | 11.7 / 18.2 / 48.8 / 7.1 / 4.0 / 11.4 |

フィールド番号 f, f+0.5 で出力されることを確認。トップの誤差増はフィールド分離で縦解像度が半分になるため(合成素材の限界)。
実機の 1080i59.94 素材での検証は未実施。

## 4. 所見

### autotrace
- 今回の 2 本では **スイング区間をほぼ追跡できなかった**。sample は f29〜53 の 25 フレームだけ、
  IMG_1095 は In/Out を切っても 8 フレーム。ボール検出・インパクト確定も両方 None。
- 元ブランチの仕様書(`docs/AUTOTRACE_SPEC.md`)は 1920x1080 横位置・29.97p・固定カメラ・ゴルファー 1 人を前提に
  `rec_0611_130128.mp4` 1 本で検証しており、縦位置スマホ素材は想定外。ダウンスイングは原理的に「弧で補間」と明記。
- 弱い区間を PCHIP で外挿するため、画面外(y=2814)まで点が飛ぶ。
- 処理は速い(GPU 6〜8 s)が、モデル読込・CUDA 初期化を含む。

### club_tracker(試作)
- 12 点すべて 15 px 以内。**ただし素材 2 本で調整した結果**であり、未知の素材での汎化は未検証。
- 試作中に踏んだ失敗と対策(今後の素材でも起きうる):
  1. 背景中央値 → フィニッシュ静止が長いと人物ごと背景になる → **3 フレーム差分**に変更
  2. レンズフレアの微動を拾う → **最小画素数 + 2 フレーム連続の一致**で追跡開始
  3. クラブの影に乗り移る → **足の最下点より下を除外**
  4. キャップ・襟・服の縁の動き → **頭部除外円 (0.2×身長代替値) + 太い人物マスク (0.17)**
  5. ダウンスイングでシャフトの残像が分断 → 連結成分ではなく **手首中心の角度ヒストグラム**(シャフトは放射状)
  6. 速い動きで MediaPipe の手首が飛ぶ → visibility < 0.5 と 1 フレーム 1/3 身長超の跳びを欠測扱い
  7. 解析幅 960 では 1080p のアイアンのシャフトが消える → **1280**
- 処理時間の 4 割が骨格推定(heavy モデル)。lite モデルや間引き(手首位置は補間で足りる)で短縮できる見込み。
- 残課題: 59.94i 実素材での検証、正面(フェースオン)素材、左打ち、複数人、カメラ微動。

## 5. 比較表

| 項目 | autotrace (YOLO) | club_tracker (骨格) |
|---|---|---|
| 精度(今回 12 点) | 評価可能 2 点、うち 1 点は破綻 | 平均 6.5 / 10.6 px、最大 14.8 px |
| 処理時間 | 6〜8 s (GPU 必須。CPU では未計測) | 32〜49 s (CPU のみ、GPU 不要) |
| 追加ディスク(開発環境) | venv 5.07 GB (torch 4.25 GB) + yolo11m-seg.pt 45 MB | mediapipe 系 約 150 MB + モデル 30 MB |
| exe への影響 | 元ブランチの spec で「数 GB」 | +148 MB(Phase 0 実測) |
| ライセンス | **ultralytics: AGPL-3.0**(商用配布は Ultralytics Enterprise License が必要)、torch: BSD-3 | mediapipe: Apache-2.0、opencv: Apache-2.0 |
| 外部ダウンロード | yolo11m-seg.pt を初回に GitHub から自動取得 | setup_assets.py で 1 回(以後オフライン) |
| 手直しの導線 | `confident_points()` / `weak_ranges()` | 同じ API 形(`confident_points()` / `weak_ranges()`)で実装 |

## 6. 判断材料

- 放送用途で exe を配布するなら、AGPL-3.0 の ultralytics を同梱する時点で法務確認が要る。
  club_tracker は Apache-2.0 のみで構成できる。
- club_tracker は 2 本の素材で合わせ込んだ試作。採用判断の前に、**実際の 1080i59.94 収録素材(できれば 5 本以上)でラベルを作って再評価**するのが先。
  `tools/label_club.py --frames` で 1 本 2〜3 分でラベルできる。
- autotrace を活かす場合も、「縦位置・スマホ素材」ではなく元ブランチの想定素材(1080p 横位置)で再評価が必要。

## 7. 再現手順

```powershell
# club_tracker
.\.venv\Scripts\python.exe tools\run_club_tracker.py <動画> build\club_eval\<名前> [--interlaced] [--debug]
# autotrace (worktree + 専用 venv)
..\golf-autotrace\.venv\Scripts\python.exe tools\run_autotrace.py ..\golf-autotrace <動画> build\club_eval\<名前> --in N --out N
# 評価
.\.venv\Scripts\python.exe tools\eval_club.py build\club_eval\<名前>\*.pred.json
```

確認動画: `build/club_eval/<名前>/{club_tracker,autotrace}_overlay.mp4`
(緑/赤丸 = 各フレームの推定、黄→赤 = 確信点のスプライン、紫十字 = 正解は `build/club_eval/frames/final_cmp_*.png`)
