"""クラブヘッド軌道の自動検出

手打ちで置いていた軌道ポイントを自動生成する。

考え方
------
姿勢推定 (キーポイント) は使わない。手首の座標が数百px 飛ぶと、そこから
決めていた探索範囲ごと壊れるため。使うのは次の5つだけ。

  1. カメラ固定 → 背景の中央値と違い、かつ前後フレームとも違う画素 = 動く物
  2. ゴルファー = 「動いている人物」。ギャラリーは止まっているので、
     人物マスクの中の動き量が最大のものを選べば確実に当たる。
     本人は消し、候補はシルエットから一定距離離れている事を要求する
  3. ヘッドはどの画素も1-2フレームしか占めない。体はゆっくりなので同じ
     画素に居座る → 前後4フレームで4回以上反応した画素を捨てる
  4. ヘッドはコンパクト → ヘッド径の円盤から3倍径の円盤を引くと、細い
     シャフトは均されて消え、ヘッドだけがピークとして立つ。
     伸びたブレはシャフト上にもピークを作るので、シルエットから遠い方を優遇
  5. 追跡は「加速度が小さく、かつ止まらない」経路を厳密DPで選ぶ

ボールが見つかれば、消えたフレーム = インパクトとしてヘッド位置を1点確定
できる。見つからなくても軌道自体は出る。

出力は TrajectoryData がそのまま食える [(x, y, frame), ...]。
"""

import cv2
import numpy as np

_SEG_MODEL = None


def _model():
    """YOLOモデルを遅延ロード (起動を遅くしないため)"""
    global _SEG_MODEL
    if _SEG_MODEL is None:
        from ultralytics import YOLO
        _SEG_MODEL = YOLO("yolo11m-seg.pt")
    return _SEG_MODEL


def _device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class AutoTraceResult:
    def __init__(self, points, impact_frame, ball_xy, conf):
        self.points = points            # [(x, y, frame), ...]
        self.impact_frame = impact_frame
        self.ball_xy = ball_xy
        self.conf = conf                # {frame: 0..}

    def control_points(self, n=14):
        """スプライン制御点に間引く (手打ち相当の点数にする)"""
        if len(self.points) <= n:
            return list(self.points)
        idx = np.linspace(0, len(self.points) - 1, n).round().astype(int)
        return [self.points[i] for i in sorted(set(idx.tolist()))]

    def confident_points(self, min_conf=1.5, max_pts=18, min_gap=2,
                         impact_guard=8):
        """信用できる点だけを制御点として返す

        インパクト周辺はヘッドが1フレームで数百px動き、証拠が薄い上に
        スコアだけ高い誤検出が出る (実測でスコア2-4の点が観客の帽子に
        乗っていた)。信頼度では弾けないので、この帯は一律で外す。
        球の位置で確定しているインパクト自身だけは残す。
        """
        guard = set()
        if self.impact_frame is not None and impact_guard > 0:
            guard = {f for f in range(self.impact_frame - impact_guard,
                                      self.impact_frame + impact_guard + 1)
                     if f != self.impact_frame}
        good = [(x, y, f) for x, y, f in self.points
                if self.conf.get(f, 0.0) >= min_conf and f not in guard]
        if len(good) < 2:
            return self.control_points(max_pts)
        out = [good[0]]
        for p in good[1:]:
            if p[2] - out[-1][2] >= min_gap:
                out.append(p)
        if out[-1][2] != good[-1][2]:
            out.append(good[-1])
        if len(out) > max_pts:
            idx = np.linspace(0, len(out) - 1, max_pts).round().astype(int)
            out = [out[i] for i in sorted(set(idx.tolist()))]
        return out

    def weak_ranges(self, min_conf=1.5, min_len=3, impact_guard=8):
        """自動では埋められない連続区間を返す [(開始, 終了), ...]

        オペレーターに「ここに点を足してください」と示すために使う。
        """
        guard = set()
        if self.impact_frame is not None and impact_guard > 0:
            guard = {f for f in range(self.impact_frame - impact_guard,
                                      self.impact_frame + impact_guard + 1)
                     if f != self.impact_frame}
        fs = [f for _, _, f in self.points]
        out, run = [], []
        for f in fs:
            if self.conf.get(f, 0.0) < min_conf or f in guard:
                run.append(f)
            else:
                if len(run) >= min_len:
                    out.append((run[0], run[-1]))
                run = []
        if len(run) >= min_len:
            out.append((run[0], run[-1]))
        return out


# ---------------------------------------------------------------- 内部処理

def _disc(d):
    k = np.zeros((d, d), np.float32)
    cv2.circle(k, (d // 2, d // 2), d // 2, 1, -1)
    return k / k.sum()


def _pass1(video_path, in_frame, out_frame, seg_every, scale, progress):
    """1パス目: 背景サンプル・セグメント用キーフレーム・球検出用フレーム

    人物マスクは毎フレーム要らない。シルエットはヘッドよりずっとゆっくり
    変わるので、数フレームおきに求めて間は前後の和で代用できる。
    ここではキーフレームの画像だけ溜めて、後でまとめて推論する
    (1枚ずつ呼ぶとオーバーヘッドが支配的になる)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    n = out_frame - in_frame + 1
    q = max(4, n // 4)
    bg_v, bg_s, keys, key_img = [], [], [], []
    vs = {}          # {frame: (V, S)} 縮小済み。2パス目の再デコードを省く
    bgr = {}         # {frame: 縮小BGR} 3枚に1枚。球検出に使う
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or i > out_frame + 1:
            break
        if i >= in_frame - 1:
            idx = i - in_frame
            if progress and idx % 20 == 0:
                progress(0.02 + 0.13 * max(idx, 0) / max(n, 1), "読み込み")
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            vs[i] = (hsv[:, :, 2].copy(), hsv[:, :, 1].copy())
            if 0 <= idx and idx % 4 == 0:
                bg_v.append(vs[i][0])
                bg_s.append(vs[i][1])
            if in_frame <= i <= out_frame and (i - in_frame) % seg_every == 0:
                keys.append(i)
                key_img.append(f.copy())
            if i % 3 == 0:
                bgr[i] = f.copy()
        i += 1
    cap.release()
    if not bg_v:
        raise IOError("フレームを読めません")
    return bg_v, bg_s, keys, key_img, bgr, vs


def _median_bg(bg_v, bg_s, stripes=8):
    """画素ごと中央値。横帯ごとに求める (全部積むと数百MBになる)"""
    h, w = bg_v[0].shape
    bv = np.empty((h, w), np.float32)
    bs = np.empty((h, w), np.float32)
    sy = (h + stripes - 1) // stripes
    for y0 in range(0, h, sy):
        y1 = min(y0 + sy, h)
        bv[y0:y1] = np.median(np.stack([a[y0:y1] for a in bg_v]), 0)
        bs[y0:y1] = np.median(np.stack([a[y0:y1] for a in bg_s]), 0)
    return bv, bs


def _golfer_masks(keys, key_img, bg_v, batch, progress):
    """キーフレームだけセグメントし、ゴルファー(=動いている人物)を選ぶ

    動き量は背景との差で代用する。前後フレームを持たなくて済むので
    キーフレーム推論と相性が良い。
    """
    seg = _model()
    dev = _device()
    h, w = bg_v.shape
    masks, dists = {}, {}
    er = np.ones((11, 11), np.uint8)
    for b0 in range(0, len(keys), batch):
        if progress:
            progress(0.15 + 0.25 * b0 / max(len(keys), 1), "人物検出")
        imgs = key_img[b0:b0 + batch]
        res = seg.predict(imgs, verbose=False, conf=0.25, classes=[0],
                          device=dev)
        for off, r in enumerate(res):
            k = keys[b0 + off]
            diff = np.abs(cv2.cvtColor(imgs[off], cv2.COLOR_BGR2HSV)[:, :, 2]
                          .astype(np.float32) - bg_v)
            body, best = np.zeros((h, w), np.uint8), -1.0
            if r.masks is not None:
                for mm in r.masks.data.cpu().numpy():
                    m = (mm > 0.5).astype(np.uint8)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
                    if m.sum() < 4000:
                        continue
                    # 平均ではなく合計。ゴルファーは体の大半がゆっくりなので、
                    # 平均だと小さく揺れたギャラリーに負ける。
                    sc = float(diff[m > 0].sum())
                    if sc > best:
                        best, body = sc, m
            masks[k] = cv2.erode(body, er)
            dists[k] = np.clip(
                cv2.distanceTransform(1 - np.clip(body, 0, 1), cv2.DIST_L2, 3),
                0, 255).astype(np.uint8)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return masks, dists


def _scan(vs, in_frame, out_frame, bg, masks, dists, keys, min_dist,
          ball, progress):
    """証拠マップを作る (1パス目で縮小済みの V/S を使い回す)

    キーフレーム間は、前後2枚のマスクの和を体とみなす。和までの距離は
    それぞれの距離マップの min で求まるので、距離変換を毎フレーム
    計算し直す必要がない。
    """
    bg_v = bg[0].astype(np.int16)
    bg_s = bg[1].astype(np.int16)
    h, w = bg_v.shape
    kar = np.array(keys)
    win, ev, raw, bdist, series = vs, {}, {}, {}, {}
    bx = by = lvl = None
    if ball is not None:
        bx, by, lvl = int(ball[0]), int(ball[1]), int(ball[2])
    n = out_frame - in_frame + 1
    for i in sorted(vs):
        if True:
            if bx is not None and in_frame <= i <= out_frame:
                patch = win[i][0][max(by - 8, 0):by + 9,
                                  max(bx - 8, 0):bx + 9]
                series[i] = int((patch > lvl).sum())
            k = i - 1
            if in_frame <= k <= out_frame and (k - 1) in win and (k + 1) in win:
                if progress and (k - in_frame) % 10 == 0:
                    progress(0.42 + 0.33 * (k - in_frame) / max(n, 1),
                             f"ヘッド検出 {k - in_frame}/{n}")
                vk, sk = win[k]
                vm, sm = win[k - 1]
                vp, sp = win[k + 1]
                # 全部 int16 で処理する。float32 にすると 2M 要素の配列を
                # 何度も舐めることになり、ここが全体の律速になる。
                # 動きの判定に明度だけを使うと、暗く色の濃い領域を通過する
                # 数フレームで落ちる。ヘッドは灰色なのでそこでは彩度の
                # 変化として出る。両方を足して判定する。
                dv1 = cv2.absdiff(vk, vm)
                dv2 = cv2.absdiff(vk, vp)
                ds1 = cv2.absdiff(sk, sm)
                ds2 = cv2.absdiff(sk, sp)
                mv = cv2.add(cv2.min(dv1, dv2).astype(np.int16),
                             (cv2.min(ds1, ds2).astype(np.int16) * 6) // 5)
                # 背景との差は符号を問わない。ヘッドのブレは半透明なので、
                # 芝の上では背景より暗く、暗いギャラリーの前では明るく写る。
                # 彩度は「落ちた分」だけ足す (色物は逆に上がる)。
                e = cv2.add(cv2.absdiff(vk.astype(np.int16), bg_v),
                            (np.maximum(bg_s - sk.astype(np.int16), 0) * 6) // 5)
                e[mv <= 20] = 0
                j = int(np.argmin(np.abs(kar - k)))
                near = [keys[j]]
                if j + 1 < len(keys):
                    near.append(keys[j + 1])
                if j > 0:
                    near.append(keys[j - 1])
                near = [x for x in near if abs(x - k) <= len(keys) and True][:2]
                body = masks[near[0]].copy()
                dist = dists[near[0]]
                for x in near[1:]:
                    body |= masks[x]
                    dist = np.minimum(dist, dists[x])
                e[body > 0] = 0
                e[dist < min_dist] = 0
                bdist[k] = dist
                ev[k] = (e > 26).astype(np.uint8)
                raw[k] = np.clip(e, 0, 255).astype(np.uint8)
    return ev, raw, bdist, series


def _candidates(ev, raw, bdist, head_d, progress=None):
    """時間的に疎な画素だけ残し、「筋の一番太い所」を候補にする

    シャフトは 2-3px しかないのに対し、ヘッドは十数px の塊として写る。
    動いている画素の内側距離変換を取ると、太い所ほど値が大きくなるので、
    そのピークがそのままヘッドになる。伸びたブレでシャフト途中を掴む
    問題も、これで原理的に起きない。
    """
    order = sorted(ev)
    cands, resp, ecl = {}, {}, {}
    dil = np.ones((21, 21), np.uint8)
    cnt = np.zeros(ev[order[0]].shape, np.uint8)
    win_lo, win_hi = order[0] - 4, order[0] - 4
    for n, k in enumerate(order):
        if progress and n % 10 == 0:
            progress(0.65 + 0.2 * n / max(len(order), 1), "候補を絞り込み")
        while win_hi <= k + 4:                 # 走査窓を差分更新する
            if win_hi in ev:
                cnt += ev[win_hi]
            win_hi += 1
        while win_lo < k - 4:
            if win_lo in ev:
                cnt -= ev[win_lo]
            win_lo += 1
        e = raw[k].astype(np.float32)
        # トップ付近ではヘッドがほぼ止まるので、しきい値を厳しくすると
        # ヘッド自身を消してしまう。体を落とすのに足りる程度に留める。
        e[cv2.dilate((cnt >= 7).astype(np.uint8),
                     np.ones((7, 7), np.uint8)) > 0] = 0
        b = cv2.morphologyEx((e > 26).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
        h, w = b.shape
        # 証拠がある範囲だけ処理する。距離変換と極大探索を毎フレーム
        # フル解像度で回すと、ここが二番目に重い処理になる。
        nz = cv2.boundingRect(b)
        inner = np.zeros((h, w), np.float32)
        if nz[2] > 0 and nz[3] > 0:
            x0 = max(nz[0] - 32, 0)
            y0 = max(nz[1] - 32, 0)
            x1 = min(nz[0] + nz[2] + 32, w)
            y1 = min(nz[1] + nz[3] + 32, h)
            sub = cv2.distanceTransform(b[y0:y1, x0:x1], cv2.DIST_L2, 5)
            inner[y0:y1, x0:x1] = cv2.GaussianBlur(sub, (5, 5), 0)
        resp[k] = np.clip(inner * 12, 0, 255).astype(np.uint8)
        ecl[k] = raw[k]
        thick = max(3.0, head_d * 0.22)
        peak = (inner >= cv2.dilate(inner, dil)) & (inner > thick)
        ys, xs = np.nonzero(peak)
        out = []
        for y, x in zip(ys, xs):
            far = float(bdist[k][y, x])
            out.append((float(x), float(y),
                        float(inner[y, x]) / max(thick, 1.0)
                        + min(far / 400.0, 0.8)))
        # 高速域ではシャフトが扇状に掃き、ヘッドは「太い塊」にならない。
        # ただしヘッドは必ずその扇の外縁を通るので、体から最も遠い側の
        # 画素群の重心を候補に足す (露光中央のヘッド位置に相当する)。
        nb, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
        for jj in range(1, nb):
            if st[jj, cv2.CC_STAT_AREA] < 400:
                continue
            ys2, xs2 = np.nonzero(lab == jj)
            dd = bdist[k][ys2, xs2]
            if dd.max() < 60:
                continue
            sel = dd >= np.quantile(dd, 0.88)
            cx, cy = float(xs2[sel].mean()), float(ys2[sel].mean())
            sc = 1.0 + min(float(dd.max()) / 400.0, 0.8)
            if all(np.hypot(cx - o[0], cy - o[1]) > head_d for o in out):
                out.append((cx, cy, sc))

        out.sort(key=lambda c: -c[2])
        # 近すぎるピークはまとめる
        keep = []
        for c in out:
            if all(np.hypot(c[0] - o[0], c[1] - o[1]) > head_d for o in keep):
                keep.append(c)
            if len(keep) >= 8:
                break
        cands[k] = keep
    return cands, resp, ecl


def _find_ball(early, late, masks, keys):
    """ティーアップされた球の位置を返す (消えるフレームは2パス目の系列で決める)

    球は「アドレス中ずっと同じ場所にある芝の上の小さな白い塊で、
    インパクト後に消えて戻らないもの」。足元は本人シルエットの最下部から取る。
    """
    if len(early) < 3 or len(late) < 3:
        return None
    em = np.median(np.stack(early), 0).astype(np.uint8)
    lm = np.median(np.stack(late), 0).astype(np.uint8)
    ge = cv2.cvtColor(em, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gl = cv2.cvtColor(lm, cv2.COLOR_BGR2GRAY).astype(np.int16)
    hsv = cv2.cvtColor(em, cv2.COLOR_BGR2HSV)
    grass = ((hsv[:, :, 0] > 25) & (hsv[:, :, 0] < 95) &
             (hsv[:, :, 1] > 40)).astype(np.uint8)
    grass = cv2.erode(cv2.morphologyEx(grass, cv2.MORPH_CLOSE,
                                       np.ones((31, 31), np.uint8)),
                      np.ones((9, 9), np.uint8))

    feet, hgt = [], []
    for k in keys[:max(2, len(keys) // 4)]:
        ys, xs = np.nonzero(masks[k])
        if len(ys) < 100:
            continue
        low = ys > np.percentile(ys, 97)
        feet.append([xs[low].mean(), ys[low].mean()])
        hgt.append(ys.max() - ys.min())
    if not feet:
        return None
    feet = np.median(feet, 0)
    reach = 0.45 * float(np.median(hgt))

    bright = ((hsv[:, :, 2] > 195) & (hsv[:, :, 1] < 70)).astype(np.uint8) * grass
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nc, lab, st, ce = cv2.connectedComponentsWithStats(bright, 8)
    best, best_sc = None, 0.0
    for j in range(1, nc):
        a = st[j, cv2.CC_STAT_AREA]
        bw, bh = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
        if not (20 <= a <= 500) or bw > 45 or bh > 45:
            continue
        if max(bw, bh) > 2.2 * min(bw, bh):
            continue
        bx, by = ce[j]
        # 足元はシルエット最下部 (靴底) なので、球はそれより少し上に写る
        if by < feet[1] - 0.55 * reach:
            continue
        d = np.hypot(bx - feet[0], by - feet[1])
        if d > reach:
            continue
        sel = lab == j
        drop = float((ge[sel] - gl[sel]).mean())
        if drop < 22:
            continue
        sc = drop * (1.0 - d / (1.15 * reach))
        if sc > best_sc:
            iy, ix = int(by), int(bx)
            lvl = int(ge[max(iy - 8, 0):iy + 9,
                         max(ix - 8, 0):ix + 9].max()) - 25
            best_sc, best = sc, (float(bx), float(by), lvl)
    return best


def _impact_from_series(series):
    """球パッチの明るさ系列から、消えて戻らない最初のフレームを返す"""
    ks = sorted(series)
    if len(ks) < 12:
        return None
    # 先頭フレームを基準にすると、まだティーアップしていない場合に
    # 「最初から消えている」と誤判定する。系列の最大値 (= 球がはっきり
    # 写っている状態) を基準にし、一度present になった後で消える所を探す。
    peak = max(series.values())
    if peak <= 0:
        return None
    present = [k for k in ks if series[k] >= 0.6 * peak]
    if not present:
        return None
    first = present[0]
    gone = {k: series[k] < max(3, peak * 0.35) for k in ks}
    tail = [k for k in ks if k > first]
    for j, k in enumerate(tail):
        window = tail[j:j + 8]
        if len(window) >= 4 and all(gone[x] for x in window):
            return k
    return None


def _track(cands, order, impact, ball, max_gap=8, w_data=1.0, w_acc=1.0,
           w_gap=0.35, w_step=0.30, w_edge=0.5, acc_scale=200.0,
           w_slow=0.020, slow_ref=25.0, time_scale=1.0,
           w_turn=0.45, w_ratio=1.6):
    """候補列から時間的に一貫したヘッド軌跡を選ぶ (厳密DP)

    状態は「直前に選んだ観測と今の観測」のペア。遷移コストは速度変化
    (= 加速度) なので、円弧を描く動きは安く、誤検出への飛び移りは高い。
    スイング中のヘッドは止まらないので、遅い経路にも罰を与える。
    始端・終端が窓を覆っていない分にもコストを掛けるので、
    「強い観測を数点だけ拾って終わり」にはならない。

    time_scale: 素材が何倍スローか。スローだと1フレームあたりの移動量が
                1/n になるので、速度・加速度の基準もそれに合わせないと
                「正しい経路が遅すぎる」と判定されて破綻する。
    """
    ts = max(time_scale, 1e-6)
    slow_ref = slow_ref / ts
    acc_scale = acc_scale / (ts * ts)
    max_gap = max(2, int(round(max_gap * ts)))
    obs = []
    for k in order:
        if impact is not None and ball is not None and k == impact:
            obs.append((k, float(ball[0]), float(ball[1]), 8.0))
            continue
        for c in cands.get(k, []):
            obs.append((k, float(c[0]), float(c[1]), float(c[2])))
    m = len(obs)
    if m < 4:
        return {}
    T = [o[0] for o in obs]
    P = [(o[1], o[2]) for o in obs]
    node = [w_step - w_data * o[3] for o in obs]
    t0, t1 = order[0], order[-1]

    # 観測はフレーム順に並んでいるので、各 j に対して許容ギャップに入る i は
    # 連続した区間になる。二重ループ (m^2 = 数十万回) を避けて範囲で取る。
    Tarr = np.array(T)
    lo = np.searchsorted(Tarr, Tarr - max_gap, "left")
    hi = np.searchsorted(Tarr, Tarr - 1, "right")
    cntp = np.maximum(hi - lo, 0)
    PJ = np.repeat(np.arange(m), cntp)
    PI = np.concatenate([np.arange(lo[j], hi[j]) for j in range(m)
                         if cntp[j] > 0]) if cntp.sum() else np.array([], int)
    pairs = list(zip(PI.tolist(), PJ.tolist()))

    def vel(i2, j2):
        dt = T[j2] - T[i2]
        return ((P[j2][0] - P[i2][0]) / dt, (P[j2][1] - P[i2][1]) / dt)

    def slow(i2, j2):
        dt = T[j2] - T[i2]
        sp = np.hypot(P[j2][0] - P[i2][0], P[j2][1] - P[i2][1]) / dt
        return w_slow * max(0.0, slow_ref - sp)

    INF = float("inf")
    npairs = len(pairs)
    if npairs == 0:
        return {}
    Tn = Tarr.astype(float)
    Px = np.array([p[0] for p in P])
    Py = np.array([p[1] for p in P])
    dt = Tn[PJ] - Tn[PI]
    vx = (Px[PJ] - Px[PI]) / dt
    vy = (Py[PJ] - Py[PI]) / dt
    sp = np.hypot(Px[PJ] - Px[PI], Py[PJ] - Py[PI]) / dt
    slow_c = w_slow * np.maximum(0.0, slow_ref - sp)
    nodec = np.array(node)
    base = nodec[PJ] + w_gap * (dt - 1) + slow_c
    cost = nodec[PI] + base + w_edge * (Tn[PI] - t0)
    back = np.full(npairs, -1, np.int64)

    # ノード単位でまとめて緩和する。ペアごとに Python ループを回すと
    # 数百万回になるので、ここは numpy で一気に計算する。
    ins, outs = [[] for _ in range(m)], [[] for _ in range(m)]
    for n2 in range(npairs):
        ins[PJ[n2]].append(n2)
        outs[PI[n2]].append(n2)
    for i2 in sorted(range(m), key=lambda x: T[x]):
        pre, suc = ins[i2], outs[i2]
        if not pre or not suc:
            continue
        pre = np.array(pre)
        suc = np.array(suc)
        # 速度ベクトルの差をそのまま使うと、速さの激変と向きの激変を
        # 区別できない。ヘッドはインパクト前後で速さが数倍になる一方、
        # 向きは滑らかに変わる。分けて評価しないと、正しい経路が
        # 「加速しすぎ」と判定されて捨てられる。
        sp_p = np.maximum(np.hypot(vx[pre], vy[pre]), 1e-3)
        sp_s = np.maximum(np.hypot(vx[suc], vy[suc]), 1e-3)
        cosang = ((vx[pre][:, None] * vx[suc][None, :]
                   + vy[pre][:, None] * vy[suc][None, :])
                  / (sp_p[:, None] * sp_s[None, :]))
        turn = np.arccos(np.clip(cosang, -1.0, 1.0))          # 向きの変化(rad)
        ratio = np.abs(np.log(sp_s[None, :] / sp_p[:, None]))  # 速さの変化(対数)
        d = turn / w_turn + ratio / w_ratio
        tot = cost[pre][:, None] + w_acc * d
        arg = np.argmin(tot, 0)
        cand = tot[arg, np.arange(len(suc))] + base[suc]
        upd = cand < cost[suc]
        if upd.any():
            cost[suc[upd]] = cand[upd]
            back[suc[upd]] = pre[arg[upd]]

    fin = cost + w_edge * (t1 - Tn[PJ])
    best_n = int(np.argmin(fin))

    chain, n = [], best_n
    while n >= 0:
        chain.append(int(PJ[n]))
        if back[n] < 0:
            chain.append(int(PI[n]))
            break
        n = int(back[n])
    chain = sorted(set(chain), key=lambda o: T[o])
    return {int(T[o]): (P[o][0], P[o][1], obs[o][3]) for o in chain}


def _refine(xs, ys, span, ecl, radius=26, min_resp=25.0, iters=3,
            only=None):
    """ブレの重心に位置を吸着させる

    center-surround の応答はブレの「縁」で立つので、そのまま使うと
    ヘッドの端に寄ってしまう。露光中央のヘッド位置はブレそのものの
    重心なので、証拠マップ (背景との差) の重心へ寄せる。
    """
    ker = np.array([1.0, 2, 3, 2, 1])
    ker /= ker.sum()
    xs, ys = np.array(xs, float), np.array(ys, float)
    for _ in range(iters):
        px = np.convolve(np.pad(xs, 2, "edge"), ker, "valid")
        py = np.convolve(np.pad(ys, 2, "edge"), ker, "valid")
        nx, ny = xs.copy(), ys.copy()
        for i, k in enumerate(span):
            if only is not None and not only[i]:
                continue          # 補間で埋めた区間は動かさない
            r = ecl.get(k)
            if r is None:
                continue
            h, w = r.shape
            x0 = max(int(px[i]) - radius, 0)
            x1 = min(int(px[i]) + radius + 1, w)
            y0 = max(int(py[i]) - radius, 0)
            y1 = min(int(py[i]) + radius + 1, h)
            sub = r[y0:y1, x0:x1].astype(np.float32)
            if sub.size == 0 or sub.max() < min_resp:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1]
            d2 = (gx - px[i]) ** 2 + (gy - py[i]) ** 2
            # 明るさで重み付けするとクラウンの反射 (ブレの上端) に
            # 引っ張られる。二値にして「形の重心」を取る。
            wt = ((sub > min_resp).astype(np.float32)
                  * np.exp(-d2 / (2 * (radius * 0.75) ** 2)))
            t = wt.sum()
            if t <= 0:
                continue
            nx[i] = float((gx * wt).sum() / t)
            ny[i] = float((gy * wt).sum() / t)
        xs, ys = nx, ny
    return xs, ys


def _fill(track, resp, order, radius=70, min_resp=6.0):
    """観測が無いフレームを、予測位置まわりの局所探索で埋める"""
    ks = sorted(track)
    if len(ks) < 2:
        return track
    xs = np.interp(order, ks, [track[k][0] for k in ks])
    ys = np.interp(order, ks, [track[k][1] for k in ks])
    out = dict(track)
    for i, k in enumerate(order):
        if k in track or not (ks[0] <= k <= ks[-1]) or k not in resp:
            continue
        r = resp[k]
        h, w = r.shape
        x0, x1 = max(int(xs[i]) - radius, 0), min(int(xs[i]) + radius, w)
        y0, y1 = max(int(ys[i]) - radius, 0), min(int(ys[i]) + radius, h)
        sub = r[y0:y1, x0:x1]
        if sub.size == 0 or sub.max() < min_resp:
            continue
        dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        out[k] = (float(x0 + dx), float(y0 + dy), 0.4)
    return out


# ---------------------------------------------------------------- 公開API

def autotrace(video_path, in_frame=0, out_frame=-1, progress=None,
              head_diameter=None, min_body_dist=None, seg_every=10, batch=4,
              scale=0.4, time_scale=1.0):
    """クラブヘッド軌道を自動検出する

    progress: callable(ratio 0..1, message) — GUIの進捗表示用
    head_diameter / min_body_dist: 省略時はゴルファーの背丈から自動決定
    seg_every: 人物マスクを何フレームおきに求めるか (大きいほど速い)。
               シルエットはヘッドよりずっとゆっくり変わるので粗くて足りる
    time_scale: 入力が何倍スロー化された素材か (2倍スローなら 2)。
                1フレームあたりの移動量が変わるので追跡の基準を合わせる
    scale: 検出を行う解像度。ヘッドは十数px あるので 0.4 でも成立し、
           画素数はフル解像度の 1/6 で済む。出力座標は元の解像度に戻す。
           0.33 まで下げるとヘッドが 5px を切って破綻する
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if out_frame < 0:
        out_frame = total - 1
    if out_frame - in_frame + 1 < 12:
        raise ValueError("フレームが足りません")

    if progress:
        progress(0.02, "読み込み")
    bg_v, bg_s, keys, key_img, bgr, vs = _pass1(
        video_path, in_frame, out_frame, seg_every, scale, progress)
    bg = _median_bg(bg_v, bg_s)
    del bg_v, bg_s
    if not keys:
        raise ValueError("解析できるフレームがありません")

    if not keys:
        raise ValueError("解析できるフレームがありません")

    masks, dists = _golfer_masks(keys, key_img, bg[0], batch, progress)
    del key_img

    # ゴルファーの背丈からスケールを決める (ユーザー設定を不要にする)
    hs = []
    for k in keys:
        ys = np.nonzero(masks[k])[0]
        if len(ys) > 100:
            hs.append(ys.max() - ys.min())
    person_h = float(np.percentile(hs, 80)) if hs else 500.0 * scale
    head_d = head_diameter or max(7, int(round(person_h / 37)) | 1)
    min_dist = min_body_dist or max(8, int(round(person_h / 18)))

    if progress:
        progress(0.40, "ボール検出")
    q = max(4, (out_frame - in_frame + 1) // 4)
    bk = sorted(k for k in bgr if in_frame <= k <= out_frame)
    early = [bgr[k] for k in bk if k < in_frame + q][:14]
    late = [bgr[k] for k in bk if k > out_frame - q][-14:]
    ball = _find_ball(early, late, masks, keys)
    del early, late, bgr
    ball_xy = (ball[0], ball[1]) if ball else None

    ev, raw, bdist, series = _scan(vs, in_frame, out_frame, bg, masks,
                                   dists, keys, min_dist, ball, progress)
    del masks, dists, vs
    if len(ev) < 8:
        raise ValueError("解析できるフレームがありません")
    impact = _impact_from_series(series) if ball else None

    cands, resp, ecl = _candidates(ev, raw, bdist, head_d, progress)
    order = sorted(cands)
    del ev, raw, bdist

    if progress:
        progress(0.90, "軌跡を追跡")
    track = _track(cands, order, impact, ball_xy, time_scale=time_scale)
    if not track:
        raise ValueError("ヘッドを追跡できませんでした")
    track = _fill(track, resp, order)

    ks = sorted(track)
    span = [k for k in order if ks[0] <= k <= ks[-1]]
    # 高速域 (ヘッドが1フレームで数百px動く区間) は証拠が薄く、拾えた点も
    # 信頼できない。そこは無理に使わず、確かな点と球の位置を通る滑らかな
    # 弧で結ぶ。手で軌道を置くときと同じ考え方。
    strong = [k for k in ks if track[k][2] >= 1.5]
    if impact is not None and ball_xy is not None and impact not in strong:
        strong = sorted(strong + [impact])
    if len(strong) >= 6:
        sx = [ball_xy[0] if (k == impact and ball_xy) else track[k][0]
              for k in strong]
        sy = [ball_xy[1] if (k == impact and ball_xy) else track[k][1]
              for k in strong]
        try:
            from scipy.interpolate import PchipInterpolator
            xs = PchipInterpolator(strong, sx)(span)
            ys = PchipInterpolator(strong, sy)(span)
        except Exception:
            xs = np.interp(span, strong, sx)
            ys = np.interp(span, strong, sy)
        for i, k in enumerate(span):
            if k in track and track[k][2] >= 1.5:
                xs[i], ys[i] = track[k][0], track[k][1]
    else:
        xs = np.interp(span, ks, [track[k][0] for k in ks])
        ys = np.interp(span, ks, [track[k][1] for k in ks])

    ker = np.array([1.0, 2, 1])
    ker /= ker.sum()
    xs = np.convolve(np.pad(xs, 1, "edge"), ker, "valid")
    ys = np.convolve(np.pad(ys, 1, "edge"), ker, "valid")
    if progress:
        progress(0.95, "位置を精密化")
    conf_ok = [(k in track and track[k][2] >= 1.5) for k in span]
    xs, ys = _refine(xs, ys, span, ecl, radius=max(12, int(26 * scale)),
                     only=conf_ok)
    xs = np.convolve(np.pad(xs, 1, "edge"), ker, "valid")
    ys = np.convolve(np.pad(ys, 1, "edge"), ker, "valid")
    if ball_xy is not None and impact in span:
        t = span.index(impact)
        xs[t], ys[t] = ball_xy

    inv = 1.0 / scale                       # 出力は元の解像度に戻す
    conf = {k: (track[k][2] if k in track else 0.0) for k in span}
    pts = [(int(round(xs[t] * inv)), int(round(ys[t] * inv)), int(k))
           for t, k in enumerate(span)]
    res = AutoTraceResult(
        pts, impact,
        (ball_xy[0] * inv, ball_xy[1] * inv) if ball_xy else None, conf)
    res.observed = {k: (track[k][0] * inv, track[k][1] * inv) for k in ks}
    res.candidates = cands
    if progress:
        progress(1.0, "完了")
    return res
