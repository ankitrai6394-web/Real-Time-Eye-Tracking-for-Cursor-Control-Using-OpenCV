"""
eye_cursor_blink_scroll.py
- Adaptive EAR calibration
- Single blink -> RIGHT CLICK
- Double blink (two short blinks within GAP) -> SCROLL DOWN
- Long blink -> SCROLL UP
- Dwell (stable gaze) -> LEFT CLICK
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import math

import numpy as np
from collections import deque

pyautogui.FAILSAFE = True  # keep for safety; move mouse to corner to abort if needed

# ---------- MediaPipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,   # gives iris/pupil indices
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# ---------- Parameters ----------
DWELL_BASE = 1.5            # base dwell time (will be adapted if you add adaptation)
DWELL_TOL_PIX = 20         # movement tolerance in pixels to consider "stable"
DOUBLE_BLINK_GAP = 0.6     # seconds between two short blinks to be considered double blink
LONG_BLINK_TIME = 0.9      # seconds eye closed => consider "long blink"
COOLDOWN_AFTER_ACTION = 0.7  # seconds cooldown so actions don't repeat rapidly

# EAR calibration default fallbacks
DEFAULT_EAR_THRESHOLD = 0.22

# landmark indices for EAR (MediaPipe 468)
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # p1..p6 ordering for our EAR function
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_PUPIL_IDX = 473
RIGHT_PUPIL_IDX = 468

# ---------- state ----------
last_pos = (0, 0)
dwell_start = None
last_action_time = 0

# blink state
eye_closed_start = None   # timestamp when eye closed began
short_blink_timestamps = deque()  # keep recent short blinks for double-blink detection

# EAR threshold (will calibrate)
ear_threshold = DEFAULT_EAR_THRESHOLD

# ---------- helper functions ----------
def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def compute_ear(landmarks, eye_idxs, img_w, img_h):
    # returns EAR robustly; uses indices as p1,p2,p3,p4,p5,p6
    pts = []
    for i in eye_idxs:
        lm = landmarks[i]
        pts.append((lm.x * img_w, lm.y * img_h))
    # ensure we have 6 points
    if len(pts) != 6:
        return 0.0
    p1,p2,p3,p4,p5,p6 = pts
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

# ---------- calibration: measure avg open EAR ----------
def calibrate_ear(duration=3.0):
    """
    Measure average EAR while user keeps eyes open.
    Returns average EAR or DEFAULT_EAR_THRESHOLD on failure.
    """
    print("[CALIBRATE] Look straight at the camera with eyes open for ~3 seconds.")
    start = time.time()
    samples = []
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try:
                ear_l = compute_ear(lm, LEFT_EYE, w, h)
                ear_r = compute_ear(lm, RIGHT_EYE, w, h)
                ear = (ear_l + ear_r) / 2.0
                samples.append(ear)
                cv2.putText(frame, f"Calibrating EAR... {len(samples)} samples", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            except Exception:
                pass
        cv2.imshow("Calibration - keep eyes open", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if len(samples) == 0:
        print("[CALIBRATE] No samples; using default EAR threshold.")
        return DEFAULT_EAR_THRESHOLD
    open_avg = float(np.median(samples))
    # choose threshold as fraction below open average
    thr = open_avg * 0.75
    print(f"[CALIBRATE] open_avg={open_avg:.3f} threshold={thr:.3f}")
    return thr

# run calibration once at start
ear_threshold = calibrate_ear(duration=3.0)

print("Calibration finished. Press 'q' to quit.")

# ---------- main loop ----------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        now = time.time()

        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark

            # pupil centers -> cursor
            try:
                lp = landmarks[LEFT_PUPIL_IDX]; rp = landmarks[RIGHT_PUPIL_IDX]
                lx, ly = int(lp.x * w), int(lp.y * h)
                rx, ry = int(rp.x * w), int(rp.y * h)
            except Exception:
                # fallback: approximate using eye bounding indices
                lx = int(landmarks[LEFT_EYE[0]].x * w); ly = int(landmarks[LEFT_EYE[0]].y * h)
                rx = int(landmarks[RIGHT_EYE[0]].x * w); ry = int(landmarks[RIGHT_EYE[0]].y * h)

            cx, cy = (lx + rx) // 2, (ly + ry) // 2
            screen_x = int((cx / w) * screen_w)
            screen_y = int((cy / h) * screen_h)

            # move mouse
            try:
                pyautogui.moveTo(screen_x, screen_y, duration=0)
            except Exception as e:
                # ignore pyautogui fails for now
                pass

            # Dwell left-click
            current_pos = (screen_x, screen_y)
            move_dist = math.hypot(current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
            if move_dist < DWELL_TOL_PIX:
                if dwell_start is None:
                    dwell_start = now
                else:
                    if now - dwell_start >= DWELL_BASE and (now - last_action_time) > COOLDOWN_AFTER_ACTION:
                        pyautogui.click()  # left click
                        last_action_time = now
                        print("[ACTION] Dwell -> LEFT click")
                        dwell_start = None
            else:
                dwell_start = None
            last_pos = current_pos

            # EAR & blink detection (time-based)
            ear_l = compute_ear(landmarks, LEFT_EYE, w, h)
            ear_r = compute_ear(landmarks, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2.0

            cv2.putText(frame, f"EAR:{ear:.3f} THR:{ear_threshold:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

            if ear < ear_threshold:
                # eye currently closed
                if eye_closed_start is None:
                    eye_closed_start = now
            else:
                # eye currently open
                if eye_closed_start is not None:
                    closed_time = now - eye_closed_start
                    eye_closed_start = None
                    # classify blink
                    if closed_time >= LONG_BLINK_TIME:
                        # LONG blink -> SCROLL UP
                        if now - last_action_time > COOLDOWN_AFTER_ACTION:
                            try:
                                pyautogui.scroll(300)  # positive typically scrolls up
                            except Exception:
                                pass
                            last_action_time = now
                            print(f"[ACTION] Long blink ({closed_time:.2f}s) -> SCROLL UP")
                        # clear short blink history
                        short_blink_timestamps.clear()
                    else:
                        # SHORT blink -> candidate for single/double
                        # handle double-blink detection
                        short_blink_timestamps.append(now)
                        # remove old timestamps beyond gap
                        while short_blink_timestamps and (now - short_blink_timestamps[0] > DOUBLE_BLINK_GAP):
                            short_blink_timestamps.popleft()
                        if len(short_blink_timestamps) >= 2:
                            # DOUBLE blink -> SCROLL DOWN
                            if now - last_action_time > COOLDOWN_AFTER_ACTION:
                                try:
                                    pyautogui.scroll(-300)  # negative scroll down
                                except Exception:
                                    pass
                                last_action_time = now
                                print("[ACTION] Double blink -> SCROLL DOWN")
                            short_blink_timestamps.clear()
                        else:
                            # SINGLE blink case: trigger right-click if not consumed by double blink in gap
                            # but to avoid accidentally reacting before a possible second blink, we can wait a short time
                            # implement a delayed action: if no second blink within DOUBLE_BLINK_GAP, then right-click
                            # For simplicity here, we'll schedule the single blink action by storing the time and performing if no second blink
                            # We implement this by checking timestamps outside this block (see below)
                            pass

            # check pending single-blink action: if there is exactly one short timestamp and it's older than gap => single blink
            if len(short_blink_timestamps) == 1:
                # if first blink older than gap and no second blink arrived, do single action
                if now - short_blink_timestamps[0] > DOUBLE_BLINK_GAP and (now - last_action_time) > COOLDOWN_AFTER_ACTION:
                    # SINGLE blink -> RIGHT CLICK
                    try:
                        pyautogui.click(button='right')
                    except Exception:
                        pass
                    last_action_time = now
                    print("[ACTION] Single blink -> RIGHT CLICK")
                    short_blink_timestamps.clear()

        else:
            # no face found; reset face-dependent timers
            dwell_start = None
            eye_closed_start = None
            # keep short_blink_timestamps (in case face reappears quickly)

        # show window
        cv2.imshow("Eye Cursor - Blink Scroll RightClick", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
