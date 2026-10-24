# 再学習サイクル レポート (clubhead_20261024_1446)

## 1. データの棚卸し

- クリップ 43 本、点 2242 点 (manual 1516, interpolated 563, auto_confirmed 5, negative 158)
- 学習に使える強ラベル (holdout 除く): 1521 点
- 撮影方向: face_on 41 本, dtl 2 本
- 画像サイズ: 1920x1080 41 本, 1080x1920 1 本, 720x1280 1 本
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
  - clip_1792799709730: 画面外の座標 4 点 (例 703,-21 F60)
  - clip_1792799709730: 切り出し外の点 14 点 (学習では除外される)
  - clip_1792801949818: 画面外の座標 4 点 (例 421,-24 F60)
  - clip_1792801949818: 切り出し外の点 5 点 (学習では除外される)
  - clip_1792803159563: 画像が無い 23 枚
  - clip_1792803180544: 同一フレームに複数の強ラベル [66]

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

WARNING: Logging before InitGoogle() is written to STDERR

W0000 00:00:1792806534.374543   15068 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806534.417398   15068 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806534.490309   19288 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

W0000 00:00:1792806534.969825   14236 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806535.011932   14236 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806535.478004    9176 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806535.520399    9176 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806535.970079    4648 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806536.010823    4648 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806536.448017   23628 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806536.488727   23628 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806536.926950    1820 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806536.967429    1820 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806537.410975   11440 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806537.454634   11440 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806537.943393    4432 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806537.983980   11308 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806538.423776   25212 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806538.464244   25212 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806538.903120    4984 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806538.943835    4984 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806539.384019   14220 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806539.424958   14220 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806539.871106    2740 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806539.911851    2740 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806540.358518    6672 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806540.399231    6672 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806540.847156   22624 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806540.887611   22624 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806541.322263   24028 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806541.363126   24028 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806541.813873   15216 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806541.854810   15216 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806542.371181   11316 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806542.414862   11316 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806542.865988    9308 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806542.906684    3172 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806543.361444   20568 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806543.402069   20568 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806543.830412   18352 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806543.870789   18352 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806544.300908   17756 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806544.341513   10328 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806544.806815   17324 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806544.847433   17324 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806545.291777   20796 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806545.332310   20796 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806545.762929   15104 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806545.803458   15104 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806546.247379    6560 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806546.288271    6560 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806546.762991    4008 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806546.804002    4008 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806547.237358   17188 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806547.278258    4520 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806547.714835   14924 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806547.755985   14924 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806548.170679   24492 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

W0000 00:00:1792806548.207631   24492 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

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

[camera] clip_1792799709730: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.67, 値 [0.67, 0.61, 0.52, 0.21])

[camera] clip_1792800181767: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.61, 値 [0.25, 0.44, 0.55, 0.61])

[camera] clip_1792800355067: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.76, 値 [0.6, 0.62, 0.66, 0.76])

[camera] clip_1792801949818: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.61, 値 [0.61, 0.58, 0.52, 0.46])

[camera] clip_1792802559221: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.73, 値 [0.73, 0.57, 0.35, 0.19])

[camera] clip_1792802584078: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.66, 値 [0.66, 0.61, 0.5, 0.3])

[camera] clip_1792802608400: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.81, 値 [0.81, 0.69, 0.63, 0.54])

[camera] clip_1792803159563: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.62, 値 [0.6, 0.44, 0.62, 0.49])

[camera] clip_1792803180544: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.71, 値 [0.71, 0.58, 0.37, 0.24])

[camera] clip_1792803199419: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.74, 値 [0.74, 0.7, 0.62, 0.55])

[camera] clip_1792803775488: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.71, 値 [0.71, 0.63, 0.48, 0.28])

[camera] clip_1792803793639: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.80, 値 [0.8, 0.73, 0.58, 0.53])

[camera] clip_1792803814350: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.69, 値 [0.69, 0.67, 0.56, 0.46])

[camera] clip_1792804390086: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.68, 値 [0.68, 0.59, 0.63, 0.52])

[camera] clip_1792804414476: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.66, 値 [0.66, 0.62, 0.57, 0.52])

[camera] clip_1792804441214: face_on  推定 (MediaPipe 肩幅/胴長 最大 0.77, 値 [0.77, 0.68, 0.56, 0.34])

保存先: datasets\clubhead

クリップ 43 本 (holdout 0 本を含む)、点の総数 2242

由来の内訳: manual 1516, interpolated 563, negative 158, auto_confirmed 5

学習に使える強ラベル (holdout 除く): 1521 点、負例 158 点

撮影方向: face_on 41, dtl 2

画像サイズ: 1920x1080 41, 1080x1920 1, 720x1280 1



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

clip_1792799709730    rec_1024_085036_30_160operator face_on         1920x1080   28  28   0  14  15  

clip_1792800181767    rec_1024_090038_84_236operator face_on         1920x1080   27  27   0  14   1  

clip_1792800355067    rec_1024_090131_166_309operator face_on         1920x1080   28  28   0  16   6  

clip_1792801949818    rec_1024_093157_78_220operator face_on         1920x1080   32  32   0  11  20  

clip_1792802559221    rec_1024_094036_200_320operator face_on         1920x1080   29  29   0   9  16  

clip_1792802584078    rec_1024_094123_96_218operator face_on         1920x1080   34  34   0  10   0  

clip_1792802608400    rec_1024_094207_61_179operator face_on         1920x1080   33  33   0   9   0  

clip_1792803159563    rec_1024_095041_70_202operator face_on         1920x1080   35  35   0  13   0  

clip_1792803180544    rec_1024_095123_93_206operator face_on         1920x1080   29  29   0   9   0  

clip_1792803199419    rec_1024_095211_110_273operator face_on         1920x1080   30  30   0  20   0  

clip_1792803775488    rec_1024_100039_33_164operator face_on         1920x1080   32  32   0  14  10  

clip_1792803793639    rec_1024_100131_27_155operator face_on         1920x1080   32  32   0  12  13  

clip_1792803814350    rec_1024_100218_169_311operator face_on         1920x1080   30  30   0  12   0  

clip_1792804390086    rec_1024_101049_217_384operator face_on         1920x1080   33  33   0  17  29  

clip_1792804414476    rec_1024_101141_381_538operator face_on         1920x1080   33  33   0  18  15  

clip_1792804441214    rec_1024_101239_111_231operator face_on         1920x1080   32  32   0  12  14  

label_IMG_1095_1080p  IMG_1095_1080p      claude   dtl             1080x1920    6   6   0  14   0  

label_sample_30fps    sample_30fps        claude   dtl             720x1280     6   6   0  18   0  



=== 前回 (inventory.json) からの増分 ===

クリップ 27 → 43 本 (+16)

点の総数 1396 → 2242 (+846)

由来の増分: auto_confirmed +0, interpolated +210, manual +497, negative +139

撮影方向の増分: dtl 2 → 2, face_on 25 → 41

強ラベル (holdout 除く): 1024 → 1521

新しいクリップ: rec_1024_085036_30_160 (face_on, 強28/負15), rec_1024_090038_84_236 (face_on, 強27/負1), rec_1024_090131_166_309 (face_on, 強28/負6), rec_1024_093157_78_220 (face_on, 強32/負20), rec_1024_094036_200_320 (face_on, 強29/負16), rec_1024_094123_96_218 (face_on, 強34/負0), rec_1024_094207_61_179 (face_on, 強33/負0), rec_1024_095041_70_202 (face_on, 強35/負0), rec_1024_095123_93_206 (face_on, 強29/負0), rec_1024_095211_110_273 (face_on, 強30/負0), rec_1024_100039_33_164 (face_on, 強32/負10), rec_1024_100131_27_155 (face_on, 強32/負13), rec_1024_100218_169_311 (face_on, 強30/負0), rec_1024_101049_217_384 (face_on, 強33/負29), rec_1024_101141_381_538 (face_on, 強33/負15), rec_1024_101239_111_231 (face_on, 強32/負14)



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

  - clip_1792799709730: 画面外の座標 4 点 (例 703,-21 F60)

  - clip_1792799709730: 切り出し外の点 14 点 (学習では除外される)

  - clip_1792801949818: 画面外の座標 4 点 (例 421,-24 F60)

  - clip_1792801949818: 切り出し外の点 5 点 (学習では除外される)

  - clip_1792803159563: 画像が無い 23 枚

  - clip_1792803180544: 同一フレームに複数の強ラベル [66]
```

## 2. 学習

- モデル: `C:\Users\yoshi\Desktop\golf-swing-broadcast\models\clubhead_20261024_1446.onnx` (0.68 MB, tiny, 169441 params)
- 学習 1296 点 (負例 158) / 検証 308 点、検証クリップ: clip_1792628587548, clip_1792630404661, clip_1792633090581, clip_1792715006688, clip_1792716178200, clip_1792803180544, clip_1792803793639, clip_1792803814350, label_IMG_1095_1080p
- 撮影方向ごとの学習/検証クリップ数: face_on 33/8, dtl 1/1
- 60 epochs, 1163.2 秒 (cuda)
- 検証精度 (元解像度 px): 平均 118.9 / 中央値 17.1 / PCK@15 0.396 / PCK@30 0.713 (検出 268/295)
- 学習側の当てはまり: 平均 27.5 px / PCK@15 0.519

| epoch | loss | 検証 平均px | 中央値px | PCK@15 | PCK@30 |
|---|---|---|---|---|---|
| 10 | 1.8417 | 169.2 | 21.2 | 0.282 | 0.606 |
| 20 | 1.3853 | 147.5 | 18.0 | 0.39 | 0.639 |
| 30 | 1.1333 | 141.9 | 18.6 | 0.37 | 0.671 |
| 40 | 0.9518 | 93.7 | 17.0 | 0.399 | 0.733 |
| 50 | 0.7699 | 111.9 | 17.0 | 0.409 | 0.722 |
| 60 | 0.6727 | 118.9 | 17.1 | 0.396 | 0.713 |

### 前回からの増分

- クリップ +16 本、点 +846、強ラベル 1024 → 1521
- 由来: auto_confirmed +0, interpolated +210, manual +497, negative +139
- 撮影方向: dtl 2 → 2, face_on 25 → 41

## 2b. 学習曲線 (学習クリップの割合を変えて学習、検証クリップは同じ)

| 割合 | 学習クリップ | 学習点 | 検証 中央値 px | 検証 PCK@15 | 検証 PCK@30 | 検出率 | モデル |
|---|---|---|---|---|---|---|---|
| 25% | 9 | 251 | 21.9 | 0.321 | 0.55 | 262/295 | `clubhead_20261024_1446_f25` |
| 50% | 17 | 531 | 18.9 | 0.392 | 0.638 | 268/295 | `clubhead_20261024_1446_f50` |
| 75% | 26 | 846 | 17.9 | 0.367 | 0.67 | 267/295 | `clubhead_20261024_1446_f75` |
| 100% | 34 | 1151 | 17.1 | 0.396 | 0.713 | 268/295 | `clubhead_20261024_1446` |

前回の学習曲線との比較 (検証クリップは今回と前回で異なるので目安):

| 割合 | 前回 学習点 | 前回 中央値 px | 前回 PCK@30 | 今回 学習点 | 今回 中央値 px | 今回 PCK@30 |
|---|---|---|---|---|---|---|
| 25% | 144 | 23.1 | 0.538 | 251 | 21.9 | 0.55 |
| 50% | 339 | 19.3 | 0.632 | 531 | 18.9 | 0.638 |
| 75% | 539 | 17.7 | 0.716 | 846 | 17.9 | 0.67 |
| 100% | 769 | 16.9 | 0.702 | 1151 | 17.1 | 0.713 |

## 2c. 軌道テンプレート (trajectory_prior) の比較 (holdout、観測を隠すマスク評価)

**20261024_1221_prior_c3_current_holdout.md** # 推定 (trajectory_prior) 評価 (20261024_1221, tag=c3_current, split=holdout, hide=0.3, block=0.25s, seed=0, prior_dir=models/)

```
× 局面 (誤差 / 身長代替値、中央値)

方式       方向       局面               n     中央値      平均
prior    dtl      down            13   0.499   0.575
prior    dtl      follow          17   0.235   0.255
prior    face_on  address-top     45   0.183   0.210
prior    face_on  down            51   0.514   0.491
prior    face_on  follow          69   0.338   0.380
物理       dtl      down            13   0.499   0.575
物理       dtl      follow          17   0.235   0.255
物理       face_on  address-top     45   0.251   0.275
物理       face_on  down            51   0.784   0.738
物理       face_on  follow          69   0.354   0.404
```
**20261024_1222_prior_c3_new_holdout.md** # 推定 (trajectory_prior) 評価 (20261024_1222, tag=c3_new, split=holdout, hide=0.3, block=0.25s, seed=0, prior_dir=models/prior_20261024_1048)

```
× 局面 (誤差 / 身長代替値、中央値)

方式       方向       局面               n     中央値      平均
prior    dtl      down            13   0.499   0.575
prior    dtl      follow          17   0.235   0.255
prior    face_on  address-top     45   0.212   0.221
prior    face_on  down            51   0.518   0.501
prior    face_on  follow          69   0.373   0.397
物理       dtl      down            13   0.499   0.575
物理       dtl      follow          17   0.235   0.255
物理       face_on  address-top     45   0.251   0.275
物理       face_on  down            51   0.784   0.738
物理       face_on  follow          69   0.354   0.404
```

## 3. holdout での比較 (tools/eval_club.py)

結果ファイル: `20261024_1220_retrain_20261024_1048_holdout_87aa30d.md`

| モード | 平均観測率 | ゲート通過 | 一致 (トップ/インパクト) | 不通過 |
|---|---|---|---|---|
| 従来 | 51% | 8 / 8 | 7 | - |
| +det | 53% | 7 / 8 | 7 | 162834 |
| +det:clubhead_20261024_1446_f25 | 59% | 5 / 8 | 4 | 130035, 162834, 122051 |
| +det:clubhead_20261024_1446_f50 | 62% | 5 / 8 | 3 | 160311, 162834, 122051 |
| +det:clubhead_20261024_1446_f75 | 68% | 6 / 8 | 5 | 162834, 122051 |
| +det:clubhead_20261024_1446 | 71% | 7 / 8 | 3 | 122051 |

| 素材 | 従来 率 / ゲート | +det 率 / ゲート | +det:clubhead_20261024_1446_f25 率 / ゲート | +det:clubhead_20261024_1446_f50 率 / ゲート | +det:clubhead_20261024_1446_f75 率 / ゲート | +det:clubhead_20261024_1446 率 / ゲート |
|---|---|---|---|---|---|---|
| 0611_130035 | 58% / OK | 62% / OK | 69% / NG | 75% / OK | 87% / OK | 88% / OK |
| 0611_150726 | 39% / OK | 39% / OK | 47% / OK | 62% / OK | 40% / OK | 44% / OK |
| 0611_151507 | 66% / OK | 75% / OK | 69% / OK | 70% / OK | 83% / OK | 69% / OK |
| 0611_160311 | 13% / OK | 13% / OK | 29% / OK | 18% / NG | 77% / OK | 76% / OK |
| 0611_162834 | 86% / OK | 88% / NG | 88% / NG | 89% / NG | 86% / NG | 88% / OK |
| 0611_163425 | 40% / OK | 40% / OK | 60% / OK | 58% / OK | 61% / OK | 88% / OK |
| 0611_165010 | 30% / OK | 30% / OK | 32% / OK | 35% / OK | 30% / OK | 34% / OK |
| 0612_122051 | 79% / OK | 79% / OK | 80% / NG | 86% / NG | 83% / NG | 81% / NG |

正解ラベル (px 誤差) は結果ファイルの「誤差」節を参照 (ラベルが無い素材は観測率・ゲートのみ)。

## 4. 判断材料 (機械的な目安)

- 新モデル vs 現行: ゲート通過 7→7、平均観測率 53%→71%、一致 7→3 → **見送り (現行のまま)**
- 従来方式 (検出器なし): ゲート通過 8、平均観測率 51%、一致 7
- 上は目安。確認動画 (`build/club_eval/real/<素材>_det_<モデル名>/club_tracker_overlay.mp4`) で改善・悪化した素材を見て決める。

## 5. 採用するときの差し替え手順 (まだ差し替えない)

```powershell
Copy-Item models\clubhead.onnx models\clubhead_prev.onnx          # 現行を退避
Copy-Item models\clubhead_20261024_1446.onnx models\clubhead.onnx            # 新モデルを現行に
Copy-Item models\clubhead_20261024_1446.train.json models\clubhead_train.json
python tools\eval_club.py --all --split tune --tag <tag> --detector both   # tune で退行確認
powershell -ExecutionPolicy Bypass -File tools\release.ps1              # exe に同梱して配布
```
戻すときは `clubhead_prev.onnx` を `clubhead.onnx` に戻す。現場 PC は `_internal\models\clubhead.onnx` を差し替えるだけでもよい。

## 6. 次回に向けて不足しているデータ

- 撮影方向の偏り: face_on 41 本, dtl 2 本。少ない方向 (特に後方 DTL と斜め) の実収録を増やす
- 同じティー・同じ背景のクリップが大半なら、別ホール・別の時間帯 (逆光・夕方)・別の服装 (白ウェア) を足す
- 負例 (人が削除した自動点) が少ないと誤検出を学べない。自動生成 → 間違いを Delete → 書き出し の手順で貯める
- HFR 2x (29.97fps 記録) とインターレース素材、ドライバー以外 (アイアン・ウェッジ) の割合

## 7. 確認動画の目視 (2026-10-24、Claude、100% モデル clubhead_20261024_1446)

| 素材 | 現行 clubhead.onnx | 新 (100%) | 判定 |
|---|---|---|---|
| 163425 (後方) | 1 区間 F78-172、ヘッドに沿う (良)、局面一致 oo | F6-64 録音スタッフの脚で静止 (ゲート止め)。F71-234 は f71 スタッフの脚、f152 だけトップのヘッド、f193/f234 地面 → 混在で通過。局面一致 xx | **悪化** (前回と同じ傾向) |
| 160311 (後方・引き) | 13%、1 区間 | F6-295 は f78/f150 芝の上の静止点、f295 フォローのヘッド (混在)、F335-351 フィニッシュ付近 (概ねヘッド)。局面一致 oo | 観測は増えたが前半は背景。判断保留 |
| 162834 (正面・看板) | ゲート NG | F6-154: f6/f43 手元付近、f80/f117 トップのヘッド、f154 バッグ → 混在で通過 | 同等 (混在のまま通過するようになった) |
| 130035 (正面・ギャラリー) | 混在で通過 | F6-259: f69 木、f132 トップのヘッド、f196 観客 → 混在 | 同等 |

改善した素材: なし (はっきり良くなった区間が無い)。悪化: 163425。

### 学習曲線の前回比較

同じ場面のデータを +500 点足しても、検証 (同じティーの未学習クリップ) の中央値は 16.9 → 17.1 px、PCK@30 は 0.70 → 0.71 で頭打ち。
holdout では観測率 75% → 71%、ゲート通過 5 → 7、一致 3 → 3 で、増えた観測は依然として別場面では人物・背景。

### 軌道テンプレートの比較 (holdout、観測 30% を 0.25 秒ブロックで隠す)

| 方向 / 局面 | 現行 (24 本 972 点) | 新 (40 本 1461 点) | 物理のみ |
|---|---|---|---|
| 正面 アドレス〜トップ | 0.183 | 0.212 | 0.251 |
| 正面 ダウンスイング | 0.514 | 0.518 | 0.784 |
| 正面 インパクト後 | 0.338 | 0.373 | 0.354 |
| 後方 (テンプレート無し) | 0.499 / 0.235 | 同じ | 同じ |

新テンプレートは改善せず (わずかに悪化)。同一場面のサンプルが増えても別コースの holdout には効かない。

### 判断

- **検出器: 採用しない** (現行 clubhead.onnx のまま)。
- **テンプレート: 採用しない** (現行 trajectory_prior_face_on.json のまま)。新テンプレートは `models/prior_20261024_1048/` に保存。
- 3 サイクル連続で「同じ 1 番ティー正面の追加」では別場面への汎化が出ない。次は **後方 (DTL) と別ホール・別背景** の実収録が無いと、
  学習・テンプレートとも改善しない。負例 (+139) が貯まり始めたのは良い兆候。
