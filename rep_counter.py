import ctypes
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from overlay import draw_overlay
import torch


# ----------------------------------------------------------------------------
# COCO 17-keypoint layout. The pose model always returns keypoints in this
# fixed order, so we refer to joints by their index.
# ----------------------------------------------------------------------------
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

HIDE_KPTS = [KP["nose"], KP["l_eye"], KP["r_eye"], KP["l_ear"], KP["r_ear"]]

# ----------------------------------------------------------------------------
# Exercise definitions.
#   joints      : the three keypoints (left side, right side) forming the angle.
#                 The angle is measured at the middle joint.
#   down_thresh : angle (deg) below which the rep is in the "down"/flexed phase.
#   up_thresh   : angle (deg) above which we count the rep as completed.
#   good_depth  : optional angle that counts as "good form" depth feedback.
# ----------------------------------------------------------------------------
EXERCISES = {
    "squat": {
        "left":  ("l_hip", "l_knee", "l_ankle"),
        "right": ("r_hip", "r_knee", "r_ankle"),
        "up_thresh": 160,
        "down_thresh": 100,
        "good_depth": 90,   # knee below 90 deg = nice deep squat
        "label": "SQUATS",
    },
    "curl": {
        "left":  ("l_shoulder", "l_elbow", "l_wrist"),
        "right": ("r_shoulder", "r_elbow", "r_wrist"),
        "down_thresh": 90,   # elbow tightly flexed = top of the curl
        "up_thresh": 140,    # arm extended = bottom
        "good_depth": 60,
        "label": "BICEP CURLS",
    },
    "pushup": {
        "left":  ("l_shoulder", "l_elbow", "l_wrist"),
        "right": ("r_shoulder", "r_elbow", "r_wrist"),
        "down_thresh": 90,
        "up_thresh": 160,
        "good_depth": 80,
        "label": "PUSH-UPS",
    },
        "shoulder_pressshoulder_press": {
        "left": ("l_shoulder", "l_elbow", "l_wrist"),
        "right": ("r_shoulder", "r_elbow", "r_wrist"),
        "down_thresh": 80,
        "up_thresh": 130,
        "good_depth": 60,
        "wrist_down_max": 0.2,  # at the bottom, wrist must NOT be overhead
        "wrist_up_min": 0.9,  # at lockout, wrist must be clearly overhead
        "label": "SHOULDER PRESS",
    }
}

CONF_THRESH = 0.5  # ignore keypoints the model isn't confident about


def calculate_angle(a, b, c):
    """Return the angle in degrees at point b, formed by segments b->a and b->c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle


def best_side_angle(kpts_xy, kpts_conf, exercise):
    """
    Compute the joint angle using whichever body side is more clearly visible.
    Returns (angle, side) or (None, None) if neither side is confident enough.
    """
    best = (None, None, -1.0)  # (angle, side, mean_confidence)
    for side in ("left", "right"):
        names = exercise[side]
        idxs = [KP[n] for n in names]
        confs = kpts_conf[idxs]
        if np.all(confs > CONF_THRESH):
            pts = [kpts_xy[i] for i in idxs]
            ang = calculate_angle(*pts)
            mean_c = float(np.mean(confs))
            if mean_c > best[2]:
                best = (ang, side, mean_c)
    return best[0], best[1]


def wrist_height(kpts_xy, kpts_conf, side):
    """Wrist height above the shoulder, in shoulder-widths (+ = above).
    Returns None if it can't be measured confidently."""

    ls, rs = KP["l_shoulder"], KP["r_shoulder"]
    sh, wr = KP[f"{side}_shoulder"], KP[f"{side}_wrist"]
    idxs = [ls, rs, sh, wr]
    if not np.all(kpts_conf[idxs] > CONF_THRESH):
        return None
    scale = float(np.linalg.norm(kpts_xy[ls] - kpts_xy[rs]))
    if scale < 1e-6:
        return None
    return float(kpts_xy[sh][1] - kpts_xy[wr][1]) / scale

def uses_wrist_gate(exercise):
    """True if the excersise defines wrist-height constraints"""
    return ("wrist_down_max" in exercise) or ("wrist_up_min" in exercise)

def hide_kp(result, indices):
    if not indices or result.keypoints is None:
        return
    data = result.keypoints.data
    if data is None or len(data) == 0 or data.shape[-1] < 3:
        return
    data = data.clone() if hasattr(data, "clone") else data.copy()
    data[:, indices, 2] = 0.0
    result.keypoints.data = data


def load_model(model_path="yolo11n-pose.pt"):
    """Load the YOLO pose model and pick a device. Meant to be called once and
    reused across sessions -- reloading it per exercise would be slow."""
    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"   # 0 = first GPU
    model.to(device)
    return model, device


def open_capture(source, width=1280, height=720, fps=30):
    """Open a webcam (digit source, e.g. 0) or a video file (path). Raises
    RuntimeError on failure instead of exiting, so callers (like a GUI) can
    handle it without killing the whole process."""
    is_webcam = str(source).isdigit()
    if is_webcam:
        cap = cv2.VideoCapture(int(source), cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    if is_webcam:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Requested {width}x{height}@{fps} -> "
              f"camera gave {actual_w}x{actual_h}@{actual_fps:.1f}")
    return cap


def _center_window(window_name, frame_w, frame_h):
    """Move an existing OpenCV window to the middle of the screen. imshow()
    itself has no concept of screen center -- it opens wherever the OS/backend
    defaults to -- so this has to be done explicitly after the window exists."""
    try:
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return
    x = max(0, (screen_w - frame_w) // 2)
    y = max(0, (screen_h - frame_h) // 2)
    cv2.moveWindow(window_name, x, y)


def run_session(exercise_key, model, device, cap, smooth=5,
                window_name="POSE (press q to quit)", flush_frames=3, mirror=True):
    """Run one live workout session for a single exercise and return the final
    rep count. Does not own `model`, `device`, or `cap` -- the caller is
    responsible for their lifecycle, so this can be called again for a
    different exercise without reloading the model or reopening the camera.

    `mirror`: flip the feed horizontally so it behaves like a mirror (left/right
    match your own movement, as in a webcam/selfie view). Leave this on for a
    live webcam; turn it off when replaying a pre-recorded video file, since
    flipping a file source would misrepresent the footage."""
    exercise = EXERCISES[exercise_key]

    # The camera may have been idle since the last session; drop a few frames
    # so the first one we show is fresh, not something buffered from before.
    for _ in range(flush_frames):
        cap.read()

    count, stage = 0, "up"
    hit_depth = False
    min_h = None              # lowest wrist height seen during the current rep
    reject_msg = None         # feedback text shown after a rejected rep
    reject_ttl = 0            # how many more frames to show it for
    bottom_seen = False
    angle_buf = deque(maxlen=smooth)
    positioned = False   # center the window once, on its first appearance

    print(f"Counting {exercise['label']}. Press 'q' in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        # ---- The CV step: run pose estimation on this frame ----
        results = model(frame, device=device, verbose=False)
        r = results[0]
        kpts_xy = kpts_conf = None
        if r.keypoints is not None and len(r.keypoints) > 0:
            kpts_xy = r.keypoints.xy[0].cpu().numpy().copy()
            kpts_conf = r.keypoints.conf[0].cpu().numpy().copy()

        hide_kp(r, HIDE_KPTS)
        # r.plot() renders the detected skeleton onto the frame for us.
        annotated = r.plot()

        angle, wrist_h, side = None, None, None
        if kpts_xy is not None:
            # The wrist gate must measure the SAME arm we
            # measured the joint angle on.
            raw_angle, side = best_side_angle(kpts_xy, kpts_conf, exercise)

            if raw_angle is not None:
                angle_buf.append(raw_angle)
                angle = float(np.mean(angle_buf))  # smoothed angle

                side_ = 'r' if side == "right" else 'l'
                wrist_h = wrist_height(kpts_xy, kpts_conf, side_)
                # print(f"wrist_height {wrist_h}")
                d_max = exercise.get("wrist_down_max")  # None -> no bottom check
                u_min = exercise.get("wrist_up_min")    # None -> no top check

                # ---- The state machine: this is what counts a rep ----
                if angle < exercise["down_thresh"]:
                    if stage != "down":
                        min_h, bottom_seen = None, False  # fresh rep
                    stage = "down"
                    if angle < exercise["good_depth"]:
                        hit_depth = True

                    # BOTTOM CHECK: remember the LOWEST wrist height reached.
                    # Testing h every frame kills the rep during the descent and
                    # ascent, when the elbow is bent but the wrist is still high.
                    if wrist_h is not None:
                        min_h = wrist_h if min_h is None else min(min_h, wrist_h)

                    # Once the wrist has reached shoulder level, this looks like
                    # a genuine descent -- only then is form feedback meaningful.
                    if d_max is None or (min_h is not None and min_h < d_max):
                        bottom_seen = True

                if angle > exercise["up_thresh"] and stage == "down":
                    stage = "up"
                    # Did the wrist ever come down to shoulder level this rep?
                    bottom_ok = (d_max is None) or (min_h is not None and min_h < d_max)
                    # Is it clearly overhead now?
                    top_ok = (u_min is None) or (wrist_h is not None and wrist_h > u_min)

                    if bottom_ok and top_ok:
                        count += 1
                    else:
                        reject_msg = ("REP REJECTED: no lockout" if not top_ok
                                      else "REP REJECTED: never lowered")
                        reject_ttl = 25
                    hit_depth = False

        # let the rejection message fade after a few frames
        if reject_ttl > 0:
            reject_ttl -= 1
        else:
            reject_msg = None

        lo, hi = exercise["down_thresh"], exercise["up_thresh"]
        rom_frac = 0.0 if angle is None else max(0.0, min(1.0, (hi - angle) / (hi - lo)))
        annotated = draw_overlay(annotated, exercise, count, stage, angle, hit_depth,
                                 rom_frac=rom_frac, reject_msg=reject_msg, wrist_h=wrist_h,
                                 kpts_xy=kpts_xy, side=side, show_tips=bottom_seen)
        cv2.imshow(window_name, annotated)
        if not positioned:
            _center_window(window_name, annotated.shape[1], annotated.shape[0])
            positioned = True
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow(window_name)
    return count
