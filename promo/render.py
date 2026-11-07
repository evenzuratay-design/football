"""Zofit Even Zur - cinematic 9:16 promo renderer.

Builds a ~30s vertical commercial from the four reference photos using
eased camera moves, warm cinematic grading, light leaks, bokeh, film grain,
a gold brand reveal and a synthesized soft piano/pad soundtrack.

Usage: python3 render.py <img_woman> <img_customer> <img_delivery> <img_cards> <fonts_dir> <out.mp4>
"""
import subprocess, sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 30.0
NF = int(DUR * FPS)
rng = np.random.default_rng(7)

img_woman, img_cust, img_deliv, img_cards, font_dir, out_path = sys.argv[1:7]


def load(path, rows=None, cols=None, up=2.0):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    if rows: a = a[rows[0]:rows[1]]
    if cols: a = a[:, cols[0]:cols[1]]
    # upscale + unsharp so camera push-ins stay crisp
    a = cv2.resize(a, None, fx=up, fy=up, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(a, (0, 0), 2.2)
    a = np.clip(a + 0.55 * (a - blur), 0, 1)
    return a, up


SRC = {
    "woman": load(img_woman, rows=(126, 1476)),
    "cust": load(img_cust, rows=(16, 1356)),
    "deliv": load(img_deliv, rows=(580, 1760)),
    "cards": load(img_cards),
}
# motion-blurred street plate for the journey shot (top-left of delivery photo)
_d, _up = SRC["deliv"]
_strip = _d[: int(430 * _up), : int(300 * _up)]          # street / trees / car only
_pano = np.concatenate([_strip, _strip[:, ::-1]] * 3, axis=1)
k = np.zeros((1, 81), np.float32); k[:] = 1 / 81
SRC["street"] = (cv2.filter2D(_pano, -1, k), _up)
# heavily blurred cards plate for the brand ending
_c = SRC["cards"][0]
SRC["cardsblur"] = (cv2.GaussianBlur(_c, (0, 0), 28), SRC["cards"][1])


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_io(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def camera(src, cam0, cam1, t, efn=ease_io):
    """cam = (cx, cy, viewport_height) in ORIGINAL (pre-upscale) crop pixels."""
    img, up = SRC[src]
    sh, sw = img.shape[:2]
    e = efn(t)
    cx, cy, vh = [(a + (b - a) * e) * up for a, b in zip(cam0, cam1)]
    vh = min(vh, sh, sw * 16 / 9)
    vw = vh * 9 / 16
    cx = min(max(cx, vw / 2), sw - vw / 2)
    cy = min(max(cy, vh / 2), sh - vh / 2)
    s = H / vh
    M = np.float32([[s, 0, -s * (cx - vw / 2)], [0, s, -s * (cy - vh / 2)]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- shots
# (start, end, src, cam_start, cam_end)
SHOTS = [
    # Scene 1 - preparing the order
    (0.0, 3.3, "woman", (540, 700, 1350), (430, 520, 820)),
    (2.8, 6.4, "cards", (540, 760, 1313), (560, 640, 900)),
    # Scene 2 - driver arrives & receives the gift box
    (5.9, 8.4, "deliv", (560, 760, 900), (520, 600, 760)),
    (7.9, 10.4, "deliv", (500, 480, 760), (470, 330, 600)),
    # Scene 3 - the journey (tracking pan across motion-blurred street)
    (9.9, 13.4, "street", (130, 215, 420), (1650, 215, 420)),
    # Scene 4 - delivery to the customer
    (12.9, 16.4, "cust", (540, 800, 1340), (520, 520, 860)),
    (15.9, 19.4, "deliv", (700, 820, 700), (520, 760, 860)),
    # Scene 5 - the big reveal
    (18.9, 21.4, "cards", (300, 420, 760), (260, 380, 560)),
    (20.9, 23.4, "cust", (520, 470, 700), (520, 480, 560)),
    (22.9, 25.6, "cards", (820, 700, 820), (780, 640, 600)),
    # Final - brand reveal on softly blurred cards
    (25.0, 30.0, "cardsblur", (560, 640, 1100), (560, 640, 1000)),
]
XF = 0.5  # crossfade length


def shot_alpha(s, t):
    a, b = s[0], s[1]
    first, last = s is SHOTS[0], s is SHOTS[-1]
    fin = 1.0 if first else ease((t - a) / XF)
    fout = 1.0 if last else ease((b - t) / XF)
    return min(fin, fout)


# ---------------------------------------------------------------- grading
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.68)) ** 2)
VIGNETTE = np.clip(1.0 - 0.55 * r ** 2.2, 0.25, 1.0)[..., None]


def grade(f):
    # warm, soft-contrast commercial look
    f = f * np.float32([1.04, 1.0, 0.92]) + np.float32([0.012, 0.006, 0.0])
    f = np.clip(f, 0, 1)
    f = f + 0.10 * (f - f * f) * (2 * f - 1) * -1  # gentle S-curve
    lum = f.mean(axis=2, keepdims=True)
    f = lum + (f - lum) * 1.06
    f = f * VIGNETTE
    # soft highlight bloom
    hi = np.clip(f - 0.72, 0, 1)
    f = f + 0.35 * cv2.GaussianBlur(hi, (0, 0), 22)
    return np.clip(f, 0, 1)


def light_leak(t, center, amp):
    """warm moving glow used to sweeten transitions."""
    cx = W * (0.2 + 0.6 * ((t * 0.37) % 1.0))
    cy = H * center
    d = ((xx - cx) / 700) ** 2 + ((yy - cy) / 900) ** 2
    g = np.exp(-d)[..., None] * np.float32([1.0, 0.62, 0.28]) * amp
    return g


LEAKS = [6.15, 10.15, 13.15, 19.15, 25.3]


# bokeh particles
NB = 46
BOK = dict(x=rng.uniform(0, W, NB), y=rng.uniform(0, H, NB), r=rng.uniform(10, 46, NB),
           vy=rng.uniform(-28, -8, NB), vx=rng.uniform(-6, 6, NB), ph=rng.uniform(0, 6.28, NB),
           a=rng.uniform(0.05, 0.16, NB))


def bokeh_layer(t, strength):
    layer = np.zeros((H, W), np.float32)
    for i in range(NB):
        x = (BOK["x"][i] + BOK["vx"][i] * t) % W
        y = (BOK["y"][i] + BOK["vy"][i] * t) % H
        a = BOK["a"][i] * (0.6 + 0.4 * math.sin(t * 1.3 + BOK["ph"][i]))
        cv2.circle(layer, (int(x), int(y)), int(BOK["r"][i]), float(a), -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 3)
    return layer[..., None] * np.float32([1.0, 0.85, 0.6]) * strength


# ---------------------------------------------------------------- brand text
BRAND = "zofit even zur"
font = ImageFont.truetype(f"{font_dir}/CG-Italic.ttf", 210)
pad = 120
TRACK = 6


def text_width(fnt):
    return sum(fnt.getlength(c) for c in BRAND) + TRACK * (len(BRAND) - 1)


scale_fit = min(1.0, 900 / text_width(font))
font = ImageFont.truetype(f"{font_dir}/CG-Italic.ttf", int(210 * scale_fit))
bb = font.getbbox(BRAND)
tw, th = int(text_width(font)), bb[3] - bb[1]
mask_img = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(mask_img)
tx, ty = (W - tw) // 2 - bb[0], (H - th) // 2 - bb[1] - 20
_x = (W - tw) / 2
for c in BRAND:  # one glyph at a time -> no "fi" ligature, exact spelling
    d.text((_x, ty), c, font=font, fill=255)
    _x += font.getlength(c) + TRACK
# ornamental hairlines above / below (no extra text)
line_y1, line_y2 = ty + bb[1] - 70, ty + bb[3] + 70
TEXT_MASK = np.asarray(mask_img).astype(np.float32) / 255.0
orn = Image.new("L", (W, H), 0)
od = ImageDraw.Draw(orn)
for ly in (line_y1, line_y2):
    od.line([(W / 2 - 230, ly), (W / 2 - 22, ly)], fill=255, width=2)
    od.line([(W / 2 + 22, ly), (W / 2 + 230, ly)], fill=255, width=2)
    od.regular_polygon((W / 2, ly, 9), 4, fill=255)
ORN_MASK = np.asarray(orn).astype(np.float32) / 255.0

# champagne-gold vertical gradient across the glyph band
top, bot = ty + bb[1], ty + bb[3]
gpos = np.clip((yy - top) / max(bot - top, 1), 0, 1)
stops = [(0.0, (0.66, 0.49, 0.22)), (0.38, (0.98, 0.90, 0.68)), (0.55, (0.93, 0.80, 0.52)),
         (1.0, (0.62, 0.44, 0.18))]
GOLD = np.zeros((H, W, 3), np.float32)
for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
    m = (gpos >= p0) & (gpos <= p1)
    u = ((gpos - p0) / (p1 - p0))[m][:, None]
    GOLD[m] = np.float32(c0) * (1 - u) + np.float32(c1) * u
ORN_GOLD = np.float32([0.86, 0.72, 0.45])


def brand_overlay(f, t):
    t0 = 25.4
    if t < t0:
        return f
    p = ease((t - t0) / 1.4)            # text fade/rise in (fully visible by ~26.8s)
    po = ease((t - t0 - 0.5) / 1.2)     # ornaments
    # rise slightly while appearing
    shift = int((1 - p) * 30)
    tm = np.roll(TEXT_MASK, shift, axis=0) * p
    om = ORN_MASK * po
    # diagonal shine sweep (passes at ~27.2s and a softer one at ~29s)
    shine = np.zeros((H, W), np.float32)
    for tc, amp in ((27.2, 0.9), (29.0, 0.5)):
        pos = (t - tc) * 1100 + W / 2
        dist = (xx + (yy - H / 2) * 0.35) - pos
        shine += amp * np.exp(-(dist / 60) ** 2)
    col = np.clip(GOLD * (1 + 0.55 * shine[..., None]), 0, 1)
    # darken background gently behind the logo for legibility
    f = f * (1 - 0.30 * p)
    # glow + soft shadow
    glow = cv2.GaussianBlur(tm, (0, 0), 18)[..., None]
    shadow = cv2.GaussianBlur(np.roll(tm, 6, axis=0), (0, 0), 6)[..., None]
    f = f * (1 - 0.45 * shadow)
    f = f + glow * np.float32([0.55, 0.40, 0.18]) * 0.55
    f = f * (1 - tm[..., None]) + col * tm[..., None]
    f = f * (1 - om[..., None]) + ORN_GOLD * om[..., None]
    return np.clip(f, 0, 1)


# ---------------------------------------------------------------- render
def frame(i):
    t = i / FPS
    acc = np.zeros((H, W, 3), np.float32)
    wsum = 0.0
    for s in SHOTS:
        if s[0] <= t <= s[1]:
            a = shot_alpha(s, t)
            if a <= 0:
                continue
            u = (t - s[0]) / (s[1] - s[0])
            img = camera(s[2], s[3], s[4], u, efn=(lambda x: x) if s[2] == "street" else ease_io)
            acc += img * a
            wsum += a
    f = acc / max(wsum, 1e-6)
    f = grade(f)
    for lt in LEAKS:
        dt = abs(t - lt)
        if dt < 0.7:
            f = 1 - (1 - f) * (1 - np.clip(light_leak(t, 0.35, 0.55 * (1 - dt / 0.7) ** 2), 0, 1))
    # bokeh during reveal + ending
    bs = ease((t - 19.0) / 1.0) * 0.6 + ease((t - 25.0) / 1.0) * 0.9
    if bs > 0:
        f = 1 - (1 - f) * (1 - np.clip(bokeh_layer(t, bs), 0, 1))
    f = brand_overlay(f, t)
    # fade in / out
    f *= ease(t / 0.6) * (1 - ease((t - (DUR - 0.5)) / 0.5) * 0.0)
    # fine film grain
    g = rng.normal(0, 0.012, (H // 2, W // 2)).astype(np.float32)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    f = np.clip(f + g, 0, 1)
    return (f * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- music
def music(path):
    sr = 48000
    n = int(sr * DUR)
    tt = np.arange(n) / sr
    out = np.zeros(n, np.float64)
    midi = lambda m: 440 * 2 ** ((m - 69) / 12)
    # Cmaj9 - Am9 - Fmaj7 - G6 (x2), 3.75s per chord
    chords = [[48, 55, 64, 67, 71, 74], [45, 52, 60, 64, 67, 71], [41, 48, 57, 60, 64, 69], [43, 50, 59, 62, 64, 67]] * 2
    cl = DUR / len(chords)
    for ci, ch in enumerate(chords):
        s0 = int(ci * cl * sr)
        L = int((cl + 1.5) * sr)
        seg = np.arange(min(L, n - s0)) / sr
        env = np.minimum(seg / 1.2, 1) * np.exp(-np.maximum(seg - cl, 0) * 2.5)
        pad = sum(np.sin(2 * np.pi * midi(m) * seg) * 0.5 + np.sin(2 * np.pi * midi(m) * 1.003 * seg) * 0.5 for m in ch[:4])
        out[s0:s0 + len(seg)] += 0.035 * pad * env
        # piano-like arpeggio, 8th notes at 96bpm
        step = 60 / 96 / 2
        k = 0
        pattern = [ch[2], ch[3], ch[4], ch[5], ch[4], ch[3]]
        while k * step < cl - 0.01:
            m = pattern[k % len(pattern)] + 12
            ps = s0 + int(k * step * sr)
            pl = int(2.2 * sr)
            pseg = np.arange(min(pl, n - ps)) / sr
            if len(pseg) == 0: break
            f0 = midi(m)
            tone = (np.sin(2 * np.pi * f0 * pseg) + 0.35 * np.sin(4 * np.pi * f0 * pseg) * np.exp(-pseg * 3)
                    + 0.12 * np.sin(6 * np.pi * f0 * pseg) * np.exp(-pseg * 5))
            penv = np.exp(-pseg * 2.4) * np.minimum(pseg / 0.004, 1)
            out[ps:ps + len(pseg)] += 0.05 * tone * penv
            k += 1
        # bass
        bseg = seg
        out[s0:s0 + len(bseg)] += 0.06 * np.sin(2 * np.pi * midi(ch[0] - 12) * bseg) * env
    # shimmer chime at brand reveal
    for j, m in enumerate([84, 88, 91, 96, 100]):
        ps = int((25.5 + j * 0.09) * sr)
        pseg = np.arange(n - ps) / sr
        out[ps:] += 0.04 * np.sin(2 * np.pi * midi(m) * pseg) * np.exp(-pseg * 1.2)
    # simple reverb (stereo decorrelated noise IR)
    irl = int(2.4 * sr)
    ir_t = np.arange(irl) / sr
    left = right = None
    chans = []
    for seed in (1, 2):
        ir = np.random.default_rng(seed).normal(0, 1, irl) * np.exp(-ir_t * 2.6)
        ir /= np.sqrt((ir ** 2).sum())
        wet = np.fft.irfft(np.fft.rfft(out, n + irl) * np.fft.rfft(ir, n + irl))[:n]
        chans.append(out * 0.75 + wet * 0.55)
    st = np.stack(chans, 1)
    st *= np.minimum(tt / 1.0, 1)[:, None]
    st *= np.clip((DUR - tt) / 2.0, 0, 1)[:, None]
    st /= np.abs(st).max() / 0.8
    pcm = (st * 32767).astype(np.int16)
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())


if __name__ == "__main__":
    wav = out_path + ".wav"
    music(wav)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                          "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                          "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-shortest",
                          "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    for i in range(NF):
        p.stdin.write(frame(i).tobytes())
        if i % 90 == 0:
            print(f"frame {i}/{NF}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", out_path)
