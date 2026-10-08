import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


KP = {
    "nose": 0,
    "l_eye": 1,       "r_eye": 2,
    "l_ear": 3,       "r_ear": 4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow": 7,     "r_elbow": 8,
    "l_wrist": 9,     "r_wrist": 10,
    "l_hip": 11,      "r_hip": 12,
    "l_knee": 13,     "r_knee": 14,
    "l_ankle": 15,    "r_ankle": 16,
}

# ============================================================================
# Polished overlay layer
# ============================================================================
THEME = {
    "card":   (22, 23, 28),   "text":  (247, 248, 250), "sub":   (150, 154, 165),
    "good":   (64, 206, 122),  "warn":  (255, 178, 62),  "bad":   (245, 84, 96),
    "accent": (79, 162, 255),  "track": (58, 60, 70),
}                                          # all RGB; bgr() flips for cv2
def bgr(c): return (int(c[2]), int(c[1]), int(c[0]))


_FONT_CACHE = {}
def get_font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    font = None
    for n in names:
        try:
            font = ImageFont.truetype(n, size); break
        except Exception:
            continue
    _FONT_CACHE[key] = font or ImageFont.load_default()
    return _FONT_CACHE[key]

def _rounded(img, x, y, w, h, r, color):
    x2, y2 = x + w, y + h
    r = int(min(r, w //2, h//2))
    cv2.rectangle(img, (x + r, y), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x, y + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x+r, y+r), (x2-r, y+r), (x+r, y2-r), (x2-r, y2-r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def card(img, x, y, w, h, radius=20, color=None, alpha=0.62, shadow=True):
    H, W = img.shape[:2]
    color = color or THEME["card"]

    # Blend only the card's bounding box (+ margin for the shadow's offset and
    # blur spread)
    pad = 50
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    lx, ly = x - x0, y - y0  # card origin within the ROI

    if shadow:
        m = np.zeros((rh, rw), np.uint8)
        _rounded(m, lx + 5, ly + 9, w, h, radius + 2, 255)
        m = cv2.GaussianBlur(m, (0, 0), 11)
        a = (m.astype(np.float32) / 255.0 * 0.55)[..., None]
        roi[:] = (roi.astype(np.float32) * (1 - a)).astype(np.uint8)

    ov = roi.copy()
    _rounded(ov, lx, ly, w, h, radius, bgr(color))
    m = np.zeros((rh, rw), np.uint8)
    _rounded(m, lx, ly, w, h, radius, 255)
    a = (m.astype(np.float32) / 255.0 * alpha)[..., None]
    roi[:] = (roi.astype(np.float32) * (1 - a) + ov.astype(np.float32) * a).astype(np.uint8)


def ring(img, cx, cy, radius, thickness, frac, color):
    frac = max(0.0, min(1.0, frac))
    cv2.circle(img, (cx, cy), radius, bgr(THEME["track"]), thickness, cv2.LINE_AA)
    if frac > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0,
                    -90, -90 + int(360 * frac), bgr(color), thickness, cv2.LINE_AA)


def highlight_joint(img, kpts_xy, joints, color):
    a, b, c = [kpts_xy[KP[n]] for n in joints]
    ip = lambda p: (int(p[0]), int(p[1]))
    for p, q in ((a, b), (b, c)):
        cv2.line(img, ip(p), ip(q), bgr(color), 7, cv2.LINE_AA)
    cv2.circle(img, ip(b), 17, bgr(color), 3, cv2.LINE_AA)
    cv2.circle(img, ip(b), 9,  bgr(color), -1, cv2.LINE_AA)

def _text_w(s, font):
    l, t, r, b = font.getbbox(s)
    return r - l

def draw_overlay(frame, exercise, count, stage, angle, hit_depth,
                 rom_frac=0.0, reject_msg=None, wrist_h=None,
                 kpts_xy=None, side=None, show_tips=True, goal=None):
    H, W = frame.shape[:2]

    if reject_msg:
        status, ck = reject_msg, "bad"
    elif stage == "down" and show_tips:
        status, ck = ("GOOD DEPTH", "good") if hit_depth else ("GO DEEPER", "warn")
    elif stage == "up":
        status, ck = "READY", "accent"
    else:
        status, ck = None, "accent"
    accent = THEME[ck]

    if kpts_xy is not None and side in ("left", "right"):
        highlight_joint(frame, kpts_xy, exercise[side], accent)

    shown_count = f"{count}/{goal}" if goal else str(count)
    cw = _text_w(shown_count, get_font(66, True))
    card(frame, 24, 24, max(300, 52 + cw + 76), 150)  # stat card; grows for long "100/100" text
    rcx, rcy, rr = W - 118, 118, 74
    ring(frame, rcx, rcy, rr, 14, rom_frac, accent)  # ROM ring

    pill_font = get_font(26, bold=True)
    if status:
        pw = _text_w(status, pill_font) + 56
        px = (W - pw) // 2
        card(frame, px, H - 90, pw, 52, radius=26, color=THEME[ck], alpha=0.92)

    # all text in one anti-aliased PIL pass
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((44, 40), exercise["label"], font=get_font(24, True), fill=THEME["text"])
    d.text((44, 78), shown_count,        font=get_font(66, True), fill=THEME["text"])
    d.text((52 + cw, 128), "REPS",       font=get_font(20), fill=THEME["sub"])
    d.text((44, 150), f"stage: {stage}", font=get_font(20), fill=THEME["sub"])
    d.text((rcx, rcy - 6), f"{int(rom_frac*100)}%", font=get_font(34, True),
           fill=THEME["text"], anchor="mm")
    d.text((rcx, rcy + 22), "ROM", font=get_font(16), fill=THEME["sub"], anchor="mm")
    if status:
        d.text((W // 2, H - 90 + 26), status, font=pill_font, fill=THEME["text"], anchor="mm")
    if wrist_h is not None:
        d.text((28, H - 34), f"h {wrist_h:+.2f}", font=get_font(16), fill=THEME["sub"])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_goal_completion(frame, exercise, goal):
    """Return a copy of `frame` blurred and dimmed, with a centered
    "goal complete" card on top. Pure function: nothing is drawn in place."""
    H, W = frame.shape[:2]

    # 1) Blur. Blurring a quarter-size copy and scaling it back up is much
    #    cheaper than a huge-kernel blur at full size, and looks just as soft.
    small = cv2.resize(frame, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 5)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2) Dim, so the white text stays readable over bright backgrounds.
    img = cv2.convertScaleAbs(img, alpha=0.5)

    # 3) Centered card (card() draws in place on the image it is given).
    cw, ch = min(560, W - 40), 330
    cx, cy = (W - cw) // 2, (H - ch) // 2
    card(img, cx, cy, cw, ch, radius=28, alpha=0.88)

    # 4) Green check-mark badge, drawn with cv2 before the PIL text pass.
    bx, by, br = W // 2, cy + 78, 42
    cv2.circle(img, (bx, by), br, bgr(THEME["good"]), -1, cv2.LINE_AA)
    check = np.array([(bx - 18, by + 2), (bx - 5, by + 15), (bx + 20, by - 13)], np.int32)
    cv2.polylines(img, [check], False, (255, 255, 255), 7, cv2.LINE_AA)

    # 5) Text, in one anti-aliased PIL pass like draw_overlay.
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((W // 2, cy + 168), "GOAL COMPLETE!", font=get_font(40, True),
           fill=THEME["good"], anchor="mm")
    d.text((W // 2, cy + 224), f"{goal} {exercise['label']}", font=get_font(28, True),
           fill=THEME["text"], anchor="mm")
    d.text((W // 2, cy + 276), "Great work!", font=get_font(22),
           fill=THEME["sub"], anchor="mm")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
