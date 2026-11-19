"""Genshin world map (RGB) -> separated layers -> point-to-point path.

Layers (all HxW): ui (removed markers/text), water, road, town, cliff, altitude (0..1, relative).
Path: Dijkstra on a directed 8-neighbour grid. Climbing costs (cliff uphill, ascent),
going down is cheap (jump / glide), so A->B and B->A can differ like in the game.

Altitude: map shading is hillshade + ground colour, so raw grayscale is NOT height.
We inpaint text and roads, take LAB lightness on land, and median-smooth it (k=31)
so per-texture noise disappears but terrace steps survive. Water = 0.
ponytail: heuristic relative height; upgrade = learn it from player trajectories / in-game altitude readouts.

Run:          python map_terrain.py map.jpg --from 440,480 --to 990,380
Tune detect:  python map_terrain.py map.jpg --tune-detect   (Save -> map_detect.json)
Tune path:    python map_terrain.py map.jpg --tune
Verify:       python map_terrain.py map.jpg --label   (blind point labels -> map.jpg.gt.json)
              python map_terrain.py map.jpg --eval    (precision / recall vs those labels)
"""
import argparse
import hashlib
import json
import os
import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROAD_REFS = ["9C946C", "B8B47D", "AAA76B", "C1B985", "99975F", "C8BE91", "878768"]
LAYERS_VERSION = 4  # bump when detection code changes so old .npz caches are ignored
COST = dict(road=1.0, terrain=2.0, water=82.0, cliff_up=120.0, cliff_down=18.0, climb=800.0)  # path tuning knobs

# detection knobs (opencv HSV: hue 0-180, sat/val 0-255). Tuned values are saved to map_detect.json and loaded here.
DETECT = dict(
    ui_sat=45, ui_max_side=100, ui_max_elong=4,       # markers: compact gray blobs (sat < ui_sat) smaller than ui_max_side px
    water_hue_lo=85, water_hue_hi=110, water_sat=60,
    road_hue_tol=3, road_sat_tol=20,                  # colour-reference match
    road_bright=14, road_desat=20, road_min_len=30,   # thin bright line match (roads on light plateau)
    road_max_halfw=7,                                 # wider blobs = beach/grass, not road
    cliff_dark=10, cliff_step=60, cliff_min_len=25,   # shadow band darkness, terrace step strength, min length
    town_sat=60, town_min_area=3000,
)
DETECT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_detect.json")
if os.path.exists(DETECT_FILE):
    DETECT.update(json.load(open(DETECT_FILE, encoding="utf-8")))

# label colours (BGR) + display names, used by every renderer
CLASSES = dict(water=((230, 130, 30), "water"), road=((60, 230, 60), "road"), town=((220, 80, 200), "town"),
               cliff=((30, 30, 240), "cliff"), ui=((0, 220, 255), "UI/text (removed)"))
SHOWN = ("water", "road", "town", "cliff")  # classes drawn by default (ui is still detected + inpainted, just not drawn)


# ---------- layers ----------
def text_mask(bgr):
    """White, thin strokes (labels, white icons) incl. their dark outline."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thin_bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k) > 40  # excludes wide bright areas (snow, temple)
    m = ((hsv[..., 2] > 200) & (hsv[..., 1] < 50) & thin_bright).astype(np.uint8) * 255
    return cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0


def ui_mask(bgr):
    """Text + map markers: small gray/white/black blobs (pins, waypoints) and orange badges ('Unknown')."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cand = ((s < DETECT["ui_sat"]) | ((h <= 25) & (s > 100) & (v > 140))).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    mx = DETECT["ui_max_side"]
    small = (st[:, 2] < mx) & (st[:, 3] < mx) & (st[:, 4] > 8)  # big gray areas (city, temple) stay
    small &= np.maximum(st[:, 2], st[:, 3]) ** 2 / np.maximum(st[:, 4], 1) < DETECT["ui_max_elong"]  # icons are compact; beaches are strips
    small[0] = False
    return text_mask(bgr) | (cv2.dilate(small[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)


def town_mask(bgr):
    """Large low-saturation areas = towns / ruins (paved, walkable). Eroded so the outer wall stays a cliff."""
    s = cv2.cvtColor(cv2.medianBlur(bgr, 9), cv2.COLOR_BGR2HSV)[..., 1]
    m = cv2.morphologyEx((s < DETECT["town_sat"]).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = (st[:, 4] >= DETECT["town_min_area"]) & (st[:, 4] < 0.1 * m.size)  # ponytail: <10% of image so UI/sea margins don't count
    big[0] = False
    return cv2.erode(big[lab].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def _thin_long(m, min_side, min_elong=8, max_half_w=None):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    side = np.maximum(st[:, 2], st[:, 3])
    keep = (side >= min_side) & (side ** 2 / np.maximum(st[:, 4], 1) >= min_elong)
    if max_half_w is not None:
        hw = np.zeros(n); np.maximum.at(hw, lab.ravel(), cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3).ravel())
        keep &= hw <= max_half_w
    keep[0] = False
    return keep[lab]


def road_mask(bgr):
    """Slide colour refs  OR  bright, less-saturated thin line (catches roads on the light plateau)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Lc = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    bright = cv2.morphologyEx(Lc, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > DETECT["road_bright"]
    desat = cv2.medianBlur(hsv[..., 1], 21).astype(np.int16) - hsv[..., 1] > DETECT["road_desat"]
    line = bright & desat & (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40)
    line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _road_refs(hsv) | _thin_long(line, min_side=DETECT["road_min_len"], max_half_w=DETECT["road_max_halfw"])


def _road_refs(hsv):
    """HSV match against the slide's road colours (hue/sat tolerance), V 35-90%."""
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    m = np.zeros(h.shape, bool)
    for hx in ROAD_REFS:
        rh, rs, _ = cv2.cvtColor(np.uint8([[[int(hx[4:], 16), int(hx[2:4], 16), int(hx[:2], 16)]]]),
                                 cv2.COLOR_BGR2HSV)[0, 0].astype(np.int16)
        dh = np.abs(h - rh); dh = np.minimum(dh, 180 - dh)
        m |= (dh <= DETECT["road_hue_tol"]) & (np.abs(s - rs) <= DETECT["road_sat_tol"])
    m &= (v >= 89) & (v <= 230)
    # roads are thin + elongated; beaches / grass patches with road colour are wide blobs
    return _thin_long(m, min_side=12, max_half_w=DETECT["road_max_halfw"])


def water_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= DETECT["water_hue_lo"]) & (hsv[..., 0] <= DETECT["water_hue_hi"]) & (hsv[..., 1] > DETECT["water_sat"])


def cliff_mask(bgr, ignore=None):
    """Cliffs on this map are drawn two ways: dark shadow bands (blackhat, up to ~10px wide)
    and sharp tone steps between terraces (gradient of median-smoothed lightness).
    Short pieces (text/icon leftovers) are dropped by length; light region borders are never dark."""
    Lc = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    band = cv2.morphologyEx(cv2.bilateralFilter(Lc, 7, 30, 5), cv2.MORPH_BLACKHAT, k) > DETECT["cliff_dark"]
    m7 = cv2.medianBlur(Lc, 7).astype(np.float32)
    step = np.hypot(cv2.Sobel(m7, cv2.CV_32F, 1, 0), cv2.Sobel(m7, cv2.CV_32F, 0, 1)) > DETECT["cliff_step"]
    m = band | step
    if ignore is not None:
        m &= ~ignore
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _thin_long(m, min_side=DETECT["cliff_min_len"], min_elong=0)


def altitude(clean, road, water, k=31):
    """Relative height 0..1 from smoothed land lightness (roads inpainted so they don't form ridges)."""
    land = cv2.inpaint(clean, cv2.dilate(road.astype(np.uint8), np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    L = cv2.cvtColor(land, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    lo, hi = np.percentile(L[~water], [2, 98])
    a = cv2.medianBlur(np.clip((L - lo) / max(hi - lo, 1), 0, 1).astype(np.float32), 5)
    a = cv2.medianBlur((a * 255).astype(np.uint8), k).astype(np.float32) / 255  # uint8 needed for k>5
    a[water] = 0
    return cv2.GaussianBlur(a, (0, 0), 2)


def detect(bgr, clean=None, ui=None, with_altitude=True):
    """All layers. Pass clean/ui to skip the UI inpaint (tuner reuses it while ui_* knobs are unchanged)."""
    if clean is None:
        ui = ui_mask(bgr)
        clean = cv2.inpaint(bgr, ui.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    road, water = road_mask(clean), water_mask(clean)
    water &= ~road
    town = town_mask(bgr)
    ignore = cv2.dilate((road | water).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # road edges / shoreline aren't cliffs
    cliff = cliff_mask(clean, ignore=ignore | town)
    L = dict(clean=clean, ui=ui, road=road, water=water, cliff=cliff, town=town)
    if with_altitude:
        L["altitude"] = altitude(clean, road, water)
    return L


layers = detect  # backwards-compatible name


# ---------- planning ----------
def base_cost(L):
    """Per-pixel walking cost from masks; built at plan time so COST edits take effect without rebuilding layers."""
    base = np.full(L["road"].shape, COST["terrain"], np.float32)
    base[L["water"]] = COST["water"]
    base[L["road"]] = COST["road"]
    return base


def _graph(L, s):
    """Directed 8-neighbour graph on an s-times downsampled grid (max-pooled so 1px cliffs survive)."""
    pool = lambda a: cv2.dilate(a.astype(np.float32), np.ones((s, s), np.uint8))[::s, ::s]
    base, cliff, alt = pool(base_cost(L)), pool(L["cliff"]) > 0, L["altitude"][::s, ::s]
    H, W = base.shape
    idx = np.arange(H * W).reshape(H, W)
    rows, cols, ws = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dy and not dx:
                continue
            ys, yd = slice(max(-dy, 0), H - max(dy, 0)), slice(max(dy, 0), H - max(-dy, 0))
            xs, xd = slice(max(-dx, 0), W - max(dx, 0)), slice(max(dx, 0), W - max(-dx, 0))
            d = np.hypot(dy, dx) * s
            g = alt[yd, xd] - alt[ys, xs]                                   # >0 = going up
            w = d * base[yd, xd] + COST["climb"] * np.maximum(g, 0)
            w += d * cliff[yd, xd] * np.where(g > 0, COST["cliff_up"], COST["cliff_down"])
            rows.append(idx[ys, xs].ravel()); cols.append(idx[yd, xd].ravel()); ws.append(w.ravel())
    G = csr_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W))
    return G, W


def plan(L, start, goal, s=2):
    """start/goal = (x, y) pixels. Returns (path Nx2 int (x,y), total cost, waypoints Mx2)."""
    G, W = _graph(L, s)
    a, b = (start[1] // s) * W + start[0] // s, (goal[1] // s) * W + goal[0] // s
    dist, pred = dijkstra(G, directed=True, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        return None, np.inf, None
    nodes = [b]
    while nodes[-1] != a:
        nodes.append(pred[nodes[-1]])
    path = np.array([(n % W * s, n // W * s) for n in nodes[::-1]], np.int32)
    wps = cv2.approxPolyDP(path.reshape(-1, 1, 2), 3, False).reshape(-1, 2)
    return path, float(dist[b]), wps


# ---------- cache / viz ----------
def _read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def build_layers(path):
    """Static map -> compute once, reuse. Cache key = image mtime + code version + DETECT values."""
    key = hashlib.md5(json.dumps(DETECT, sort_keys=True).encode()).hexdigest()[:8]
    cache = path + f".layers.v{LAYERS_VERSION}.{key}.npz"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return dict(np.load(cache))
    L = detect(_read(path))
    np.savez_compressed(cache, **L)
    return L


def render_labels(L, show=SHOWN, base="clean", alpha=0.6, legend=True):
    """Every detected class in its own colour + legend with pixel share, so misses / false hits are easy to spot."""
    out = L[base].astype(np.float32) * 0.85 if base in L else L["clean"].astype(np.float32) * 0.85
    for k in show:  # drawn in CLASSES order: water < road < town < cliff < ui
        if k in L:
            m = L[k].astype(bool)
            out[m] = out[m] * (1 - alpha) + np.float32(CLASSES[k][0]) * alpha
    out = out.astype(np.uint8)
    if legend:
        rows = [k for k in show if k in L]
        x0, y0 = 10, out.shape[0] - 10 - 24 * len(rows)
        cv2.rectangle(out, (x0 - 5, y0 - 8), (x0 + 250, out.shape[0] - 4), (25, 25, 25), -1)
        for i, k in enumerate(rows):
            y = y0 + 24 * i
            cv2.rectangle(out, (x0, y), (x0 + 18, y + 16), CLASSES[k][0], -1)
            pct = 100 * L[k].mean()
            cv2.putText(out, f"{CLASSES[k][1]}  {pct:.1f}%", (x0 + 26, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render(L, path=None, wps=None):
    """Labels + optional path (yellow) with waypoints (magenta)."""
    out = render_labels(L, show=("road", "cliff"), alpha=0.65, legend=False)
    if path is not None:
        cv2.polylines(out, [path.reshape(-1, 1, 2)], False, (0, 255, 255), 2)
        for p in wps:
            cv2.circle(out, tuple(int(v) for v in p), 4, (255, 0, 255), -1)
    return out


def save_layers(L, prefix):
    w = lambda name, img: cv2.imencode(".png", img)[1].tofile(f"{prefix}_{name}.png")
    for k in CLASSES:
        w(k, L[k].astype(np.uint8) * 255)
    w("altitude", cv2.applyColorMap((L["altitude"] * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    w("labels", render_labels(L))


# ---------- interactive tuners (matplotlib; opencv here is headless) ----------
DETECT_RANGES = dict(  # slider (min, max); all integers
    ui_sat=(0, 120), ui_max_side=(20, 200), ui_max_elong=(1, 30),
    water_hue_lo=(60, 110), water_hue_hi=(90, 140), water_sat=(0, 200),
    road_hue_tol=(0, 15), road_sat_tol=(0, 60), road_bright=(2, 60), road_desat=(0, 80),
    road_min_len=(5, 120), road_max_halfw=(2, 20),
    cliff_dark=(2, 60), cliff_step=(10, 200), cliff_min_len=(5, 120),
    town_sat=(10, 150), town_min_area=(500, 20000),
)


DETECT_KO = dict(  # slider labels: what each knob means (↑ = effect of raising it)
    ui_sat="마커: 회색 기준 채도 ↑더 많이 지움", ui_max_side="마커: 최대 크기(px)", ui_max_elong="마커: 최대 길쭉함",
    water_hue_lo="물: 색상 하한", water_hue_hi="물: 색상 상한", water_sat="물: 최소 채도 ↑물 줄어듦",
    road_hue_tol="길: 기준색 색상 허용폭 ↑길 늘어남", road_sat_tol="길: 기준색 채도 허용폭 ↑길 늘어남",
    road_bright="길(밝은 선): 밝기차 ↑길 줄어듦", road_desat="길(밝은 선): 채도차 ↑길 줄어듦",
    road_min_len="길: 최소 길이(px)", road_max_halfw="길: 최대 반폭(px) ↑넓은 덩어리 허용",
    cliff_dark="절벽: 그림자 어두움 ↑절벽 줄어듦", cliff_step="절벽: 단차 세기 ↑절벽 줄어듦",
    cliff_min_len="절벽: 최소 길이(px)", town_sat="마을: 회색 기준 채도 ↑마을 늘어남", town_min_area="마을: 최소 면적(px)",
)


def _korean_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in have] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ---------- ground truth: blind point labels -> precision / recall ----------
GT_CLASSES = ("water", "road", "town", "cliff", "land")  # land = plain walkable ground (none of the others)
GT_KO = dict(water="물", road="길", town="마을", cliff="절벽", land="일반 땅")
THIN_TOL = dict(road=3, cliff=3)  # px: a point within this distance of a detected line counts as on it


def pred_class_map(L):
    """Per-pixel predicted class index into GT_CLASSES (later classes win; thin classes widened by THIN_TOL)."""
    p = np.full(L["road"].shape, GT_CLASSES.index("land"), np.uint8)
    for k in ("water", "town", "road", "cliff"):
        m = L[k].astype(np.uint8)
        if k in THIN_TOL:
            m = cv2.dilate(m, np.ones((2 * THIN_TOL[k] + 1,) * 2, np.uint8))
        p[m > 0] = GT_CLASSES.index(k)
    return p


def sample_points(L, per_class=40, seed=0):
    """Stratified by predicted class so rare classes (road, cliff) get enough points.
    w = pixels in stratum / points drawn -> weighted metrics stay unbiased after you retune DETECT."""
    p, rng, pts = pred_class_map(L), np.random.default_rng(seed), []
    for i, k in enumerate(GT_CLASSES):
        ys, xs = np.nonzero(p == i)
        n = min(per_class * (2 if k == "land" else 1), len(ys))
        for j in (rng.choice(len(ys), n, replace=False) if n else []):
            pts.append(dict(x=int(xs[j]), y=int(ys[j]), w=len(ys) / n, label=None))
    rng.shuffle(pts)
    return pts


def evaluate(L, gt):
    """Weighted precision / recall per class + accuracy + confusion counts, from labelled points ('skip' ignored)."""
    p = pred_class_map(L)
    rows = [g for g in gt if g["label"] in GT_CLASSES]
    if not rows:
        return None
    pred = np.array([GT_CLASSES[p[g["y"], g["x"]]] for g in rows])
    true = np.array([g["label"] for g in rows])
    w = np.array([g["w"] for g in rows])
    res = {}
    for k in GT_CLASSES:
        tp, pp, tt = w[(pred == k) & (true == k)].sum(), w[pred == k].sum(), w[true == k].sum()
        res[k] = dict(precision=tp / pp if pp else float("nan"), recall=tp / tt if tt else float("nan"),
                      n=int((true == k).sum()))
    res["accuracy"] = w[pred == true].sum() / w.sum()
    res["n"] = len(rows)
    res["confusion"] = {(t, q): int(((true == t) & (pred == q)).sum()) for t in GT_CLASSES for q in GT_CLASSES}
    return res


def format_eval(res):
    if res is None:
        return "정답 라벨 없음 (--label 로 먼저 라벨링)"
    lines = [f"정답 점 {res['n']}개, 정확도 {100 * res['accuracy']:.0f}%", "클래스   정밀도  재현율  (점수)"]
    for k in GT_CLASSES:
        r = res[k]
        f = lambda v: "  -  " if v != v else f"{100 * v:4.0f}%"
        lines.append(f"{GT_KO[k]:<5} {f(r['precision'])}  {f(r['recall'])}  ({r['n']})")
    return "\n".join(lines)


def print_eval(res):
    print(format_eval(res))
    if res:
        print("\n혼동행렬 (행=정답, 열=검출):")
        print("        " + "".join(f"{GT_KO[q]:>7}" for q in GT_CLASSES))
        for t in GT_CLASSES:
            print(f"{GT_KO[t]:<6}" + "".join(f"{res['confusion'][(t, q)]:>8}" for q in GT_CLASSES))


def label_tool(bgr, gt_path, show=True, radius=40):
    """Blind labelling: one random point at a time, detection hidden. Keys 1-5 = class, x = unsure/skip,
    b = back, close window = stop. Saved after every key, so you can stop and resume any time."""
    import matplotlib.pyplot as plt
    _korean_font()
    for k in list(plt.rcParams):  # free up keys like 's', 'q', 'b' from matplotlib defaults
        if k.startswith("keymap."):
            plt.rcParams[k] = []
    if os.path.exists(gt_path):
        gt = json.load(open(gt_path, encoding="utf-8"))
    else:
        gt = sample_points(detect(bgr, with_altitude=False))
        json.dump(gt, open(gt_path, "w", encoding="utf-8"))
    st = dict(i=next((i for i, g in enumerate(gt) if g["label"] is None), len(gt)))
    rgb = bgr[..., ::-1]
    pad = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw=dict(width_ratios=[1.4, 1]))
    keys = {str(i + 1): k for i, k in enumerate(GT_CLASSES)}
    help_txt = "  ".join(f"[{n}] {GT_KO[k]}" for n, k in keys.items()) + "   [x] 모르겠음   [b] 이전"

    def draw():
        ax0.clear(); ax1.clear(); ax0.axis("off"); ax1.axis("off")
        done = sum(g["label"] is not None for g in gt)
        if st["i"] >= len(gt):
            ax1.set_title(f"끝! {done}/{len(gt)}개 라벨링 완료. 창을 닫으세요."); fig.canvas.draw_idle(); return
        g = gt[st["i"]]
        ax0.imshow(rgb); ax0.plot(g["x"], g["y"], "o", ms=14, mfc="none", mec="red", mew=2.5)
        ax0.set_title(f"{st['i'] + 1}/{len(gt)}  (완료 {done})")
        ax1.imshow(pad[g["y"]:g["y"] + 2 * radius + 1, g["x"]:g["x"] + 2 * radius + 1], interpolation="nearest")
        c = radius
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # crosshair with a gap so the pixel itself stays visible
            ax1.plot([c + dx * 4, c + dx * 14], [c + dy * 4, c + dy * 14], color="red", lw=2)
        prev = g["label"]
        ax1.set_title(f"빨간 십자 중심은 무엇인가요?" + (f"  (현재: {GT_KO.get(prev, prev)})" if prev else ""))
        fig.suptitle(help_txt + f"\n선 위 판정은 중심에서 {THIN_TOL['road']}px 이내면 길/절벽", fontsize=11)
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in keys or ev.key == "x":
            if st["i"] < len(gt):
                gt[st["i"]]["label"] = keys.get(ev.key, "skip")
                json.dump(gt, open(gt_path, "w", encoding="utf-8"))
                st["i"] += 1
        elif ev.key in ("b", "backspace"):
            st["i"] = max(st["i"] - 1, 0)
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    if show:
        plt.show()
    return fig, on_key, gt


def tune_detect(bgr, show=True, out_png=None, gt=None):
    """Sliders = DETECT knobs, checkboxes = which classes to draw. Recomputes 0.3s after you stop dragging.
    With ground-truth points (gt) it shows live precision / recall, so you can see if a change actually helps.
    'Save' writes map_detect.json (used by every later run) and a labelled PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons, Button
    _korean_font()
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0.01, 0.36, 0.76, 0.62]); ax.axis("off")
    keys = list(DETECT_RANGES)
    sl = {}
    for i, k in enumerate(keys):  # two columns of sliders under the image
        col, row = divmod(i, (len(keys) + 1) // 2)
        lo, hi = DETECT_RANGES[k]
        sl[k] = Slider(fig.add_axes([0.20 + col * 0.42, 0.31 - row * 0.034, 0.18, 0.022]), DETECT_KO[k], lo, hi,
                       valinit=DETECT[k], valstep=1 if hi - lo < 1000 else 100)
        sl[k].label.set_fontsize(9)
    names = list(SHOWN)
    chk = CheckButtons(fig.add_axes([0.79, 0.78, 0.20, 0.18]), [GT_KO[k] for k in names], [True] * len(names))
    for lbl, k in zip(chk.labels, names):
        b, g, r = CLASSES[k][0]; lbl.set_color((r / 255, g / 255, b / 255)); lbl.set_fontweight("bold")
    bview = Button(fig.add_axes([0.79, 0.72, 0.20, 0.04]), "원본 / 마커 지운 지도 전환")
    bsave = Button(fig.add_axes([0.79, 0.67, 0.20, 0.04]), "저장 (json + png)")
    status = fig.text(0.79, 0.65, "", fontsize=9, va="top")
    state = dict(ui_key=None, L=None, base="clean", dirty=True)
    im = ax.imshow(bgr[..., ::-1])

    def recompute():
        DETECT.update({k: int(s.val) for k, s in sl.items()})
        ui_key = (DETECT["ui_sat"], DETECT["ui_max_side"], DETECT["ui_max_elong"])
        if ui_key != state["ui_key"]:  # inpaint only when UI knobs change (~0.5s)
            ui = ui_mask(bgr)
            state["clean"] = cv2.inpaint(bgr, ui.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
            state["ui"], state["ui_key"] = ui, ui_key
        L = detect(bgr, clean=state["clean"], ui=state["ui"], with_altitude=False)
        L["orig"] = bgr
        state["L"] = L
        print("DETECT =", dict(DETECT))
        draw()

    def draw():
        L = state["L"]
        shown = [k for k, on in zip(names, chk.get_status()) if on]
        im.set_data(render_labels(L, show=shown, base="orig" if state["base"] == "orig" else "clean")[..., ::-1])
        share = "지도 면적 비율: " + ", ".join(f"{GT_KO[k]} {100 * L[k].mean():.1f}%" for k in names)
        status.set_text(share + "\n\n" + (format_eval(evaluate(L, gt)) if gt else "정답 라벨 없음\n(--label 로 먼저 라벨링하면\n여기에 정밀도/재현율이 떠요)"))
        ax.set_title(f"바탕: {'원본' if state['base'] == 'orig' else '마커 지운 지도'}  |  슬라이더를 멈추면 0.3초 뒤 다시 계산")
        fig.canvas.draw_idle()

    def mark_dirty(_=None):
        state["dirty"] = True

    def tick():
        if state["dirty"]:
            state["dirty"] = False
            recompute()

    def toggle_view(_):
        state["base"] = "orig" if state["base"] == "clean" else "clean"
        draw()

    def save(_):
        json.dump(DETECT, open(DETECT_FILE, "w", encoding="utf-8"), indent=1)
        if out_png:
            cv2.imencode(".png", render_labels(state["L"]))[1].tofile(out_png)
        status.set_text(status.get_text() + f"\n\n저장됨 -> {os.path.basename(DETECT_FILE)}")
        print("saved", DETECT_FILE, out_png or "")
        fig.canvas.draw_idle()

    for s in sl.values():
        s.on_changed(mark_dirty)
    chk.on_clicked(lambda _: draw())
    bview.on_clicked(toggle_view); bsave.on_clicked(save)
    timer = fig.canvas.new_timer(interval=300); timer.add_callback(tick); timer.start()
    fig._keep = (sl, chk, bview, bsave, timer)  # keep widgets alive
    tick()
    if show:
        plt.show()
    return fig, state, sl, chk, save


def tune(L, show=True):
    """Path tuner: left-click = start, right-click = goal, sliders = COST. Every replan prints the current COST."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    ranges = dict(terrain=(1, 10), water=(1, 300), cliff_up=(1, 500), cliff_down=(1, 200), climb=(0, 3000))
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0.02, 0.28, 0.96, 0.7]); ax.axis("off")
    im = ax.imshow(render(L)[..., ::-1])
    sl = {k: Slider(fig.add_axes([0.15, 0.22 - i * 0.045, 0.7, 0.03]), k, lo, hi, valinit=COST[k])
          for i, (k, (lo, hi)) in enumerate(ranges.items())}
    pts = {}

    def redraw(_=None):
        COST.update({k: s.val for k, s in sl.items()})
        img = render(L)
        title = "left-click = start, right-click = goal"
        if "s" in pts and "g" in pts:
            path, cost, wps = plan(L, pts["s"], pts["g"])
            if path is not None:
                img = render(L, path, wps)
            title = f"cost={cost:.0f}  waypoints={0 if wps is None else len(wps)}"
            print("COST =", {k: round(v, 2) for k, v in COST.items()}, "|", title)
        for k, c in (("s", (0, 255, 0)), ("g", (0, 0, 255))):
            if k in pts:
                cv2.circle(img, pts[k], 7, c, 2)
        im.set_data(img[..., ::-1]); ax.set_title(title); fig.canvas.draw_idle()

    def click(ev):
        if ev.inaxes is ax and ev.button in (1, 3):
            pts["s" if ev.button == 1 else "g"] = (int(ev.xdata), int(ev.ydata))
            redraw()

    for s in sl.values():
        s.on_changed(redraw)
    fig.canvas.mpl_connect("button_press_event", click)
    if show:
        plt.show()
    return fig, click, redraw, pts


def _selfcheck():
    img = np.full((200, 300, 3), (70, 140, 110), np.uint8)              # green terrain
    cv2.line(img, (20, 150), (280, 120), (25, 50, 40), 2)              # dark cliff stroke
    cv2.line(img, (20, 30), (280, 60), (190, 215, 215), 2)             # light region border
    cv2.putText(img, "Woods", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 4)
    cv2.putText(img, "Woods", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    L = detect(img)
    assert L["cliff"][130:145, 40:260].any(), "cliff missed"
    assert not L["cliff"][25:65, 20:280].any(), "light border marked as cliff"
    assert not L["cliff"][70:105, 85:200].any(), "text marked as cliff"
    # directional: plateau (high) above a cliff wall with one ramp; up must cost more than down
    z = np.zeros((100, 100), bool)
    L = dict(road=z, water=z, cliff=z.copy(), altitude=np.zeros((100, 100), np.float32))
    L["altitude"][:50] = 1.0; L["cliff"][49:51, :] = True
    _, up, _ = plan(L, (50, 80), (50, 20))
    _, down, _ = plan(L, (50, 20), (50, 80))
    assert up > down * 3, (up, down)
    t0 = COST["terrain"]; COST["terrain"] = t0 * 2
    _, down2, _ = plan(L, (50, 20), (50, 80))
    COST["terrain"] = t0
    assert down2 > down, "COST change not applied at plan time"
    # eval: labels equal to predictions -> 100%; flipping every cliff label to land -> cliff precision 0
    L = dict(road=z, water=z, town=z, cliff=L["cliff"])
    gt = sample_points(L, per_class=10)
    p = pred_class_map(L)
    for g in gt:
        g["label"] = GT_CLASSES[p[g["y"], g["x"]]]
    assert evaluate(L, gt)["accuracy"] == 1.0
    for g in gt:
        g["label"] = "land" if g["label"] == "cliff" else g["label"]
    assert evaluate(L, gt)["cliff"]["precision"] == 0
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("map", nargs="?")
    ap.add_argument("--from", dest="src"); ap.add_argument("--to", dest="dst")
    ap.add_argument("--tune", action="store_true", help="interactive path-cost tuner")
    ap.add_argument("--tune-detect", action="store_true", help="interactive detection tuner (saves map_detect.json)")
    ap.add_argument("--label", action="store_true", help="blind ground-truth labelling (saves <map>.gt.json)")
    ap.add_argument("--eval", action="store_true", help="precision / recall of current DETECT against <map>.gt.json")
    a = ap.parse_args()
    _selfcheck()
    if a.map:
        prefix = os.path.splitext(a.map)[0]
        gt_path = a.map + ".gt.json"
        gt = json.load(open(gt_path, encoding="utf-8")) if os.path.exists(gt_path) else None
        if a.label:
            label_tool(_read(a.map), gt_path)
            raise SystemExit
        if a.tune_detect:
            tune_detect(_read(a.map), out_png=prefix + "_labels.png", gt=gt)
            raise SystemExit
        L = build_layers(a.map)
        if a.eval:
            print_eval(evaluate(L, gt) if gt else None)
            raise SystemExit
        if a.tune:
            tune(L)
            raise SystemExit
        save_layers(L, prefix)
        if a.src and a.dst:
            s, g = (tuple(map(int, v.split(","))) for v in (a.src, a.dst))
            path, cost, wps = plan(L, s, g)
            print(f"cost={cost:.0f} waypoints={wps.tolist() if wps is not None else None}")
            if path is not None:
                cv2.imencode(".png", render(L, path, wps))[1].tofile(prefix + "_path.png")
