# 再学習サイクル レポート (clubhead_20261023_982)

## 1. データの棚卸し

- クリップ 27 本、点 1396 点 (manual 1019, interpolated 353, auto_confirmed 5, negative 19)
- 学習に使える強ラベル (holdout 除く): 1024 点
- 撮影方向: face_on 25 本, dtl 2 本
- 画像サイズ: 1920x1080 25 本, 1080x1920 1 本, 720x1280 1 本
- 要確認:
  - clip_1792628620875: 同一フレームに複数の強ラベル [108, 131]
  - clip_1792628673233: 画面外の座標 1 点 (例 901,-29 F96)
  - clip_1792628673233: 切り出し外の点 1 点 (学習では除外される)
  - clip_1792631561135: 画面外の座標 1 点 (例 914,-10 F84)
  - clip_1792631561135: 切り出し外の点 1 点 (学習では除外される)
  - clip_1792714437533: 画面外の座標 4 点 (例 590,-16 F66)
  - clip_1792714437533: 切り出し外の点 8 点 (学習では除外される)
  - clip_1792714983788: 画像が無い 48 枚
  - clip_1792715006688: 画面外の座標 1 点 (例 743,-9 F108)
  - clip_1792715006688: 切り出し外の点 4 点 (学習では除外される)
  - clip_1792715025548: 切り出し外の点 1 点 (学習では除外される)

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

WARNING: Logging before InitGoogle() is written to STDERR

W0000 00:00:1792718706.363289   19312 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718706.458947   34516 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718706.540714   26404 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

W0000 00:00:1792718707.092418   13004 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718707.146056   13004 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718707.637862   24664 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718707.692852   24664 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718708.208394   29740 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718708.259259   29144 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718708.745096   16204 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718708.798034   16204 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718709.306157   18204 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718709.357809   18204 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718709.854234   22516 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718709.907266   22516 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718710.413593   18580 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718710.462573   18580 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718710.963190   31276 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718711.021060   31276 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718711.544070   19400 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718711.591267   19400 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718712.073430   23624 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718712.126977   23624 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718712.659898   26684 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718712.721084   33992 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718713.252838   31948 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792718713.305489   31948 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

[camera] clip_1792631615756: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.68, 値 [0.68, 0.59, 0.32, 0.26])

[camera] clip_1792714437533: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.78, 値 [0.78, 0.72, 0.64, 0.5])

[camera] clip_1792715006688: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.61, 値 [0.61, 0.57, 0.44, 0.25])

[camera] clip_1792715025548: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.70, 値 [0.7, 0.64, 0.6, 0.57])

[camera] clip_1792715538075: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.68, 値 [0.68, 0.53, 0.25, 0.14])

[camera] clip_1792715563445: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.66, 値 [0.66, 0.58, 0.48, 0.34])

[camera] clip_1792715585113: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.77, 値 [0.77, 0.67, 0.57, 0.42])

[camera] clip_1792716160120: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.69, 値 [0.69, 0.64, 0.54, 0.47])

[camera] clip_1792716178200: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.58, 値 [0.58, 0.58, 0.48, 0.4])

[camera] clip_1792716198524: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.70, 値 [0.7, 0.65, 0.54, 0.37])

[camera] clip_1792716778703: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.72, 値 [0.72, 0.69, 0.55, 0.36])

[camera] clip_1792716796332: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.65, 値 [0.65, 0.57, 0.3, 0.17])

[camera] clip_1792716822394: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.74, 値 [0.74, 0.6, 0.53, 0.41])

保存先: datasets\clubhead

クリップ 27 本 (holdout 0 本を含む)、点の総数 1396

由来の内訳: manual 1019, interpolated 353, negative 19, auto_confirmed 5

学習に使える強ラベル (holdout 除く): 1024 点、負例 19 点

撮影方向: face_on 25, dtl 2

画像サイズ: 1920x1080 25, 1080x1920 1, 720x1280 1



clip_id               素材                  labeler  方向              サイズ          強   手   自   補   負  備考

clip_1792628587548    rec_1022_092043_155_333operator face_on         1920x1080   54  54   0  11   0  

clip_1792628620875    rec_1022_092215_335_472operator face_on         1920x1080   48  43   5  16   0  

clip_1792628673233    rec_1022_092131_224_348operator face_on         1920x1080   39  39   0  10   0  

clip_1792629179673    rec_1022_093040_262_428operator face_on         1920x1080   59  59   0  15   0  

clip_1792629209331    rec_1022_093128_260_420operator face_on         1920x1080   51  51   0   6   0  

clip_1792629778029    rec_1022_094118_199_333operator face_on         1920x1080   39  39   0  13   0  

clip_1792630383666    rec_1022_095032_16_154operator face_on         1920x1080   44  44   0   9   0  

clip_1792630404661    rec_1022_095115_25_172operator face_on         1920x1080   38  38   0  12   0  

clip_1792631164045    rec_1022_100123_138_280operator face_on         1920x1080   39  39   0  15   0  

clip_1792631561135    rec_1022_101205_243_414operator face_on         1920x1080   49  49   0  16   0  

clip_1792631615756    rec_1022_101126_238_386operator face_on         1920x1080   46  46   0  11   0  

clip_1792633090581    rec_1022_103733_117_279operator face_on         1920x1080   40  40   0  15   0  

clip_1792714437533    rec_1023_091321_104_278operator face_on         1920x1080   37  37   0  20   0  

clip_1792714983788    rec_1023_092045_299_458operator face_on         1920x1080   32  32   0  18   7  

clip_1792715006688    rec_1023_092135_215_350operator face_on         1920x1080   37  37   0  14   0  

clip_1792715025548    rec_1023_092221_82_222operator face_on         1920x1080   38  38   0  13   0  

clip_1792715538075    rec_1023_093036_79_187operator face_on         1920x1080   42  42   0  10   0  

clip_1792715563445    rec_1023_093117_210_358operator face_on         1920x1080   39  39   0  15   0  

clip_1792715585113    rec_1023_093154_42_175operator face_on         1920x1080   41  41   0  10   0  

clip_1792716160120    rec_1023_094047_110_232operator face_on         1920x1080   37  37   0  11   0  

clip_1792716178200    rec_1023_094127_45_176operator face_on         1920x1080   34  34   0  13   0  

clip_1792716198524    rec_1023_094210_160_326operator face_on         1920x1080   29  29   0  20   0  

clip_1792716778703    rec_1023_095049_21_141operator face_on         1920x1080   35  35   0   6   6  

clip_1792716796332    rec_1023_095137_0_119operator face_on         1920x1080   34  34   0  10   6  

clip_1792716822394    rec_1023_095232_308_443operator face_on         1920x1080   31  31   0  12   0  

label_IMG_1095_1080p  IMG_1095_1080p      claude   dtl             1080x1920    6   6   0  14   0  

label_sample_30fps    sample_30fps        claude   dtl             720x1280     6   6   0  18   0  



=== 前回 (inventory.json) からの増分 ===

クリップ 13 → 27 本 (+14)

点の総数 682 → 1396 (+714)

由来の増分: auto_confirmed +0, interpolated +183, manual +512, negative +19

撮影方向の増分: dtl 2 → 2, face_on 11 → 25

強ラベル (holdout 除く): 512 → 1024

新しいクリップ: rec_1022_101126_238_386 (face_on, 強46/負0), rec_1023_091321_104_278 (face_on, 強37/負0), rec_1023_092045_299_458 (face_on, 強32/負7), rec_1023_092135_215_350 (face_on, 強37/負0), rec_1023_092221_82_222 (face_on, 強38/負0), rec_1023_093036_79_187 (face_on, 強42/負0), rec_1023_093117_210_358 (face_on, 強39/負0), rec_1023_093154_42_175 (face_on, 強41/負0), rec_1023_094047_110_232 (face_on, 強37/負0), rec_1023_094127_45_176 (face_on, 強34/負0), rec_1023_094210_160_326 (face_on, 強29/負0), rec_1023_095049_21_141 (face_on, 強35/負6), rec_1023_095137_0_119 (face_on, 強34/負6), rec_1023_095232_308_443 (face_on, 強31/負0)



要確認:

  - clip_1792628620875: 同一フレームに複数の強ラベル [108, 131]

  - clip_1792628673233: 画面外の座標 1 点 (例 901,-29 F96)

  - clip_1792628673233: 切り出し外の点 1 点 (学習では除外される)

  - clip_1792631561135: 画面外の座標 1 点 (例 914,-10 F84)

  - clip_1792631561135: 切り出し外の点 1 点 (学習では除外される)

  - clip_1792714437533: 画面外の座標 4 点 (例 590,-16 F66)

  - clip_1792714437533: 切り出し外の点 8 点 (学習では除外される)

  - clip_1792714983788: 画像が無い 48 枚

  - clip_1792715006688: 画面外の座標 1 点 (例 743,-9 F108)

  - clip_1792715006688: 切り出し外の点 4 点 (学習では除外される)

  - clip_1792715025548: 切り出し外の点 1 点 (学習では除外される)
```

## 2. 学習

- モデル: `C:\Users\yoshi\Desktop\golf-swing-broadcast\models\clubhead_20261023_982.onnx` (0.68 MB, tiny, 169441 params)
- 学習 782 点 (負例 19) / 検証 219 点、検証クリップ: clip_1792628587548, clip_1792630383666, clip_1792715006688, clip_1792715585113, clip_1792716796332, label_IMG_1095_1080p
- 撮影方向ごとの学習/検証クリップ数: face_on 20/5, dtl 1/1
- 60 epochs, 759.4 秒 (cuda)
- 検証精度 (元解像度 px): 平均 88.8 / 中央値 16.9 / PCK@15 0.383 / PCK@30 0.702 (検出 188/213)
- 学習側の当てはまり: 平均 30.9 px / PCK@15 0.517

| epoch | loss | 検証 平均px | 中央値px | PCK@15 | PCK@30 |
|---|---|---|---|---|---|
| 10 | 1.9018 | 210.7 | 17.9 | 0.344 | 0.615 |
| 20 | 1.4693 | 146.6 | 16.7 | 0.381 | 0.69 |
| 30 | 1.2424 | 129.0 | 19.2 | 0.26 | 0.657 |
| 40 | 0.9898 | 100.9 | 16.3 | 0.427 | 0.725 |
| 50 | 0.7882 | 89.2 | 17.6 | 0.324 | 0.709 |
| 60 | 0.7368 | 88.8 | 16.9 | 0.383 | 0.702 |

### 前回からの増分

- クリップ +14 本、点 +714、強ラベル 512 → 1024
- 由来: auto_confirmed +0, interpolated +183, manual +512, negative +19
- 撮影方向: dtl 2 → 2, face_on 11 → 25

## 2b. 学習曲線 (学習クリップの割合を変えて学習、検証クリップは同じ)

| 割合 | 学習クリップ | 学習点 | 検証 中央値 px | 検証 PCK@15 | 検証 PCK@30 | 検出率 | モデル |
|---|---|---|---|---|---|---|---|
| 25% | 6 | 144 | 23.1 | 0.364 | 0.538 | 195/213 | `clubhead_20261023_982_f25` |
| 50% | 11 | 339 | 19.3 | 0.363 | 0.632 | 182/213 | `clubhead_20261023_982_f50` |
| 75% | 16 | 539 | 17.7 | 0.388 | 0.716 | 183/213 | `clubhead_20261023_982_f75` |
| 100% | 21 | 769 | 16.9 | 0.383 | 0.702 | 188/213 | `clubhead_20261023_982` |

## 3. holdout での比較 (tools/eval_club.py)

結果ファイル: `20261023_1148_retrain_20261023_1025_holdout_4d07e80.md`

| モード | 平均観測率 | ゲート通過 | 一致 (トップ/インパクト) | 不通過 |
|---|---|---|---|---|
| 従来 | 51% | 8 / 8 | 7 | - |
| +det | 53% | 7 / 8 | 7 | 162834 |
| +det:clubhead_20261023_982_f25 | 59% | 6 / 8 | 6 | 162834, 122051 |
| +det:clubhead_20261023_982_f50 | 67% | 7 / 8 | 2 | 122051 |
| +det:clubhead_20261023_982_f75 | 71% | 5 / 8 | 5 | 130035, 162834, 122051 |
| +det:clubhead_20261023_982 | 75% | 5 / 8 | 3 | 130035, 162834, 122051 |

| 素材 | 従来 率 / ゲート | +det 率 / ゲート | +det:clubhead_20261023_982_f25 率 / ゲート | +det:clubhead_20261023_982_f50 率 / ゲート | +det:clubhead_20261023_982_f75 率 / ゲート | +det:clubhead_20261023_982 率 / ゲート |
|---|---|---|---|---|---|---|
| 0611_130035 | 58% / OK | 62% / OK | 66% / OK | 88% / OK | 88% / NG | 63% / NG |
| 0611_150726 | 39% / OK | 39% / OK | 42% / OK | 44% / OK | 42% / OK | 50% / OK |
| 0611_151507 | 66% / OK | 75% / OK | 70% / OK | 81% / OK | 83% / OK | 82% / OK |
| 0611_160311 | 13% / OK | 13% / OK | 13% / OK | 46% / OK | 65% / OK | 84% / OK |
| 0611_162834 | 86% / OK | 88% / NG | 84% / NG | 88% / OK | 88% / NG | 88% / NG |
| 0611_163425 | 40% / OK | 40% / OK | 80% / OK | 68% / OK | 71% / OK | 67% / OK |
| 0611_165010 | 30% / OK | 30% / OK | 30% / OK | 34% / OK | 40% / OK | 78% / OK |
| 0612_122051 | 79% / OK | 79% / OK | 90% / NG | 88% / NG | 90% / NG | 85% / NG |

正解ラベル (px 誤差) は結果ファイルの「誤差」節を参照 (ラベルが無い素材は観測率・ゲートのみ)。

## 4. 判断材料 (機械的な目安)

- 新モデル vs 現行: ゲート通過 7→5、平均観測率 53%→75%、一致 7→3 → **見送り (現行のまま)**
- 従来方式 (検出器なし): ゲート通過 8、平均観測率 51%、一致 7
- 上は目安。確認動画 (`build/club_eval/real/<素材>_det_<モデル名>/club_tracker_overlay.mp4`) で改善・悪化した素材を見て決める。

## 5. 採用するときの差し替え手順 (まだ差し替えない)

```powershell
Copy-Item models\clubhead.onnx models\clubhead_prev.onnx          # 現行を退避
Copy-Item models\clubhead_20261023_982.onnx models\clubhead.onnx            # 新モデルを現行に
Copy-Item models\clubhead_20261023_982.train.json models\clubhead_train.json
python tools\eval_club.py --all --split tune --tag <tag> --detector both   # tune で退行確認
powershell -ExecutionPolicy Bypass -File tools\release.ps1              # exe に同梱して配布
```
戻すときは `clubhead_prev.onnx` を `clubhead.onnx` に戻す。現場 PC は `_internal\models\clubhead.onnx` を差し替えるだけでもよい。

## 6. 次回に向けて不足しているデータ

- 撮影方向の偏り: face_on 25 本, dtl 2 本。少ない方向 (特に後方 DTL と斜め) の実収録を増やす
- 同じティー・同じ背景のクリップが大半なら、別ホール・別の時間帯 (逆光・夕方)・別の服装 (白ウェア) を足す
- 負例 (人が削除した自動点) が少ないと誤検出を学べない。自動生成 → 間違いを Delete → 書き出し の手順で貯める
- HFR 2x (29.97fps 記録) とインターレース素材、ドライバー以外 (アイアン・ウェッジ) の割合

## 7. 確認動画の目視 (2026-10-23、Claude、100% モデル clubhead_20261023_982)

| 素材 | 現行 clubhead.onnx | 新 (100%) | 判定 |
|---|---|---|---|
| 163425 (後方) | 1 区間 F78-172、ヘッドに沿う (良) | 7 区間。アドレス中の 4 区間はボール位置のヘッド (静止でゲート止め)、F60-192 は f93/f159 で録音スタッフの腕と顔、f126 だけヘッド。F198-234 もスタッフ上で通過 | **悪化** (スタッフを拾った区間がゲートを通る) |
| 150726 (後方) | F246-404 良 | F9-119 植え込みで静止 (ゲート止め)、F246-415 良 | 変化なし |
| 130035 (正面・ギャラリー) | 混在区間が通過 | F8-258 混在 (木・観客・バッグ、f133 だけトップのヘッド) → ゲート止め | 同等 (どちらも不安定) |
| 122051 (正面・ギャラリー) | 冒頭区間が通過 (mixed) | 1 区間 F6-169 観客・バッグ → ゲート止め | 同等 |
| 162834 (正面・看板) | ゲート NG | F6-154 混在 (帽子・トップのヘッド・バッグ) → ゲート NG | 同等 |

観測率は 53% → 75% に上がるが、増えた観測は別場面では人物・背景。改善した素材はなく、163425 が悪化。
holdout に正解ラベル (px) が無いため、px 誤差は学習データ内の検証クリップ (同じティーの未学習クリップ) でしか出せない。

### 学習曲線の見立て

検証 (同じ場面の未学習クリップ) は 25% → 100% で中央値 23 → 17 px、PCK@30 0.54 → 0.70 と改善が続くが 75% → 100% で頭打ち。
holdout (別コース) では量を増やすほど観測率は上がるがゲート通過は 7 → 5 と下がり、一致も落ちる。
**同じ場面のデータを増やしても別場面への汎化は改善しない**。次は量より場面の多様性。

### 判断: 採用しない (現行のまま)

### 次の現場で優先して集めたい条件

1. **後方 (DTL)**: 学習 2 本 12 点のみ。holdout の後方素材 (163425、150726) で悪化・無変化。後方の実収録を最優先
2. **別のホール・別の背景**: 2 日とも 1 番ティー正面の同一画角 (25 本)。看板・観客・キャディ・録音スタッフが近い場面、木や植え込みが背景の場面
3. **局面**: アドレス〜テークバックとインパクト前後は十分。トップ〜ダウンスイング (シャフトが短く見える) とフィニッシュの点を増やす
4. **環境**: 逆光・夕方・曇天、白ウェア、女子以外、HFR 2x (29.97fps) 素材
5. **負例**: 19 点 (2 本) のみ。自動生成の誤り (観客の顔・脚・植え込み) を Delete してから書き出す運用を続ける
