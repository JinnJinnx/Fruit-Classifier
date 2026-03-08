"""
Fruit Detector — CV Mini Project
Author : Jin Somaang (67011125)

Usage:
  python fruit_classifier.py <image_path>
  python fruit_classifier.py --webcam
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from collections import deque

# ── MUST match train_model.py exactly ────────────────────────
CLASSES    = ["apple", "banana", "blueberry", "guava", "orange", "pear", "strawberry"]
IMG_SIZE   = 128
MODEL_PATH = "fruit_model.keras"
CONF_THRESHOLD = 0.65   # minimum confidence to show a detection
NMS_IOU        = 0.35   # overlap threshold for removing duplicate boxes
SMOOTH_N       = 10     # webcam: average over this many frames

model = None
try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded: {MODEL_PATH}")
        print(f"[OK] Classes: {CLASSES}")
        print(f"[OK] IMG_SIZE: {IMG_SIZE}")
    else:
        print(f"[WARN] '{MODEL_PATH}' not found — run train_model.py first.")
except ImportError:
    print("[WARN] TensorFlow not installed.")


# ── Colour per fruit (BGR) ────────────────────────────────────
LABEL_COLOUR = {
    "apple":      (34,   34,  200),   # red
    "banana":     (0,   210,  255),   # yellow
    "blueberry":  (160,  40,   80),   # dark purple
    "guava":      (0,   180,   80),   # green
    "orange":     (0,   140,  255),   # orange
    "pear":       (0,   210,  160),   # lime green
    "strawberry": (60,   60,  210),   # crimson
    "unknown":    (130, 130,  130),   # grey
}


# ──────────────────────────────────────────────
# Classify one image patch
# Returns (label, confidence 0-1)
# ──────────────────────────────────────────────
def classify_patch(patch: np.ndarray) -> tuple:
    if model is None:
        return "unknown", 0.0
    rgb   = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    batch = np.expand_dims(small.astype("float32"), 0)
    probs = model.predict(batch, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = CLASSES[idx] if conf >= CONF_THRESHOLD else "unknown"
    return label, round(conf, 3)


# ──────────────────────────────────────────────
# Find fruit candidate regions via colour masks
# ──────────────────────────────────────────────
COLOUR_MASKS = [
    (np.array([0,   70,  50]),  np.array([10,  255, 255])),   # red lo
    (np.array([160, 70,  50]),  np.array([179, 255, 255])),   # red hi
    (np.array([10,  70,  70]),  np.array([35,  255, 255])),   # orange/yellow
    (np.array([35,  50,  50]),  np.array([85,  255, 255])),   # green
    (np.array([85,  50,  30]),  np.array([140, 255, 255])),   # blue/purple
    (np.array([140, 40,  30]),  np.array([160, 255, 255])),   # pink/purple
]

def find_candidate_regions(img: np.ndarray) -> list:
    H, W = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = np.zeros((H, W), dtype=np.uint8)
    for lo, hi in COLOUR_MASKS:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes    = []
    min_area = H * W * 0.005
    max_area = H * W * 0.95

    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        pad = 20
        x   = max(x - pad, 0)
        y   = max(y - pad, 0)
        w   = min(w + pad * 2, W - x)
        h   = min(h + pad * 2, H - y)
        boxes.append((x, y, w, h))

    return boxes


# ──────────────────────────────────────────────
# Non-max suppression
# ──────────────────────────────────────────────
def nms(boxes, scores, threshold=NMS_IOU) -> list:
    if not boxes:
        return []
    boxes_arr  = np.array(boxes, dtype=float)
    scores_arr = np.array(scores, dtype=float)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < threshold]
    return keep


# ──────────────────────────────────────────────
# Draw detection boxes on image
# ──────────────────────────────────────────────
def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    """detections: list of (label, conf, x, y, w, h)"""
    out = img.copy()
    H, W = out.shape[:2]

    for label, conf, x, y, w, h in detections:
        colour = LABEL_COLOUR.get(label, LABEL_COLOUR["unknown"])
        x2, y2 = min(x + w, W), min(y + h, H)

        # Transparent fill
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

        # Border
        cv2.rectangle(out, (x, y), (x2, y2), colour, 2)

        # Corner accents
        cl, ct = 18, 3
        for px, py, dx, dy in [
            (x,  y,   1,  1), (x2, y,  -1,  1),
            (x,  y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(out, (px, py), (px + dx * cl, py), colour, ct)
            cv2.line(out, (px, py), (px, py + dy * cl), colour, ct)

        # Label tag: "APPLE  0.93"
        tag = f"{label.upper()}  {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tag_y = max(y - 8, th + 6)
        cv2.rectangle(out, (x, tag_y - th - 6),
                      (x + tw + 12, tag_y + bl + 2), colour, -1)
        cv2.putText(out, tag, (x + 6, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    cv2.LINE_AA)

    # Bottom summary bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, H - 44), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    if detections:
        summary = "  |  ".join(
            f"{d[0].upper()} {d[1]:.2f}" for d in detections
        )
        summary = f"Detected: {summary}"
    else:
        summary = "No fruits detected"

    cv2.putText(out, summary, (12, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────
# Full detection pipeline
# ──────────────────────────────────────────────
def detect_fruits(img: np.ndarray) -> tuple:
    H, W   = img.shape[:2]
    scale  = 800 / max(H, W)
    proc   = cv2.resize(img, (int(W * scale), int(H * scale)),
                        interpolation=cv2.INTER_LANCZOS4)
    pH, pW = proc.shape[:2]

    candidates = find_candidate_regions(proc)

    # Fallback: use whole image if no regions found
    if not candidates:
        candidates = [(10, 10, pW - 20, pH - 20)]

    boxes, scores, labels = [], [], []
    for (x, y, w, h) in candidates:
        patch = proc[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        label, conf = classify_patch(patch)
        if label == "unknown":
            continue
        boxes.append((x, y, w, h))
        scores.append(conf)
        labels.append(label)

    keep = nms(boxes, scores)

    inv  = 1.0 / scale
    dets = [
        (labels[i], scores[i],
         int(boxes[i][0] * inv), int(boxes[i][1] * inv),
         int(boxes[i][2] * inv), int(boxes[i][3] * inv))
        for i in keep
    ]

    return draw_detections(img, dets), dets


# ──────────────────────────────────────────────
# Process uploaded image
# ──────────────────────────────────────────────
def process_image(path: str):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot open: {path}")
        return

    print(f"\nProcessing: {path}")
    annotated, dets = detect_fruits(img)

    if dets:
        print(f"Found {len(dets)} fruit(s):")
        for label, conf, x, y, w, h in dets:
            print(f"  {label.upper():<12} confidence: {conf:.3f}   box: ({x},{y},{w},{h})")
    else:
        print("No fruits detected.")

    out_path = str(Path(path).stem) + "_detected.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Result saved: {out_path}")

    cv2.namedWindow("Fruit Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fruit Detector", 1000, 700)
    cv2.imshow("Fruit Detector", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Webcam mode
# ──────────────────────────────────────────────
def run_webcam(camera_index: int = 0):
    if model is None:
        print("ERROR: No model — run train_model.py first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Fruit Detector — Real Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fruit Detector — Real Time", 1000, 650)

    print("Webcam started. Place fruit in the box. Q=quit S=save.")

    prob_buffer    = deque(maxlen=SMOOTH_N)
    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        cx   = int(W * 0.25)
        cy   = int(H * 0.15)
        cw   = int(W * 0.50)
        ch   = int(H * 0.70)
        crop = frame[cy:cy+ch, cx:cx+cw]

        rgb   = cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)),
                             cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb.astype("float32"), 0)
        probs = model.predict(batch, verbose=0)[0]
        prob_buffer.append(probs)

        avg   = np.mean(prob_buffer, axis=0)
        idx   = int(np.argmax(avg))
        conf  = float(avg[idx])
        label = CLASSES[idx] if conf >= CONF_THRESHOLD else "unknown"
        colour = LABEL_COLOUR.get(label, LABEL_COLOUR["unknown"])

        out = frame.copy()
        cv2.rectangle(out, (cx, cy), (cx+cw, cy+ch), colour, 2)

        cl, ct = 22, 3
        for px, py, dx, dy in [
            (cx,    cy,      1,  1), (cx+cw, cy,    -1,  1),
            (cx,    cy+ch,   1, -1), (cx+cw, cy+ch, -1, -1),
        ]:
            cv2.line(out, (px, py), (px + dx*cl, py), colour, ct)
            cv2.line(out, (px, py), (px, py + dy*cl), colour, ct)

        if label != "unknown":
            tag = f"{label.upper()}  {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(out, (cx, cy - th - 12),
                          (cx + tw + 12, cy), colour, -1)
            cv2.putText(out, tag, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(out, "Place fruit in box",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1,
                        cv2.LINE_AA)

        ov = out.copy()
        cv2.rectangle(ov, (0, H-50), (W, H), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.75, out, 0.25, 0, out)

        status = f"{label.upper()}  {conf:.2f}" if label != "unknown" else "Waiting..."
        cv2.putText(out, status, (14, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)

        bm = 14
        cv2.rectangle(out, (bm, H-10), (W-bm, H-4), (70, 70, 70), -1)
        cv2.rectangle(out, (bm, H-10),
                      (bm + int((W - bm*2) * conf), H-4), colour, -1)
        cv2.putText(out, "Q: quit   S: save snapshot",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Fruit Detector — Real Time", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snapshot_count += 1
            fname = f"snapshot_{snapshot_count:03d}_{label}.jpg"
            cv2.imwrite(fname, out)
            print(f"Saved -> {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--webcam":
        run_webcam(int(sys.argv[2]) if len(sys.argv) >= 3 else 0)
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fruit_classifier.py <image_path>")
        print("  python fruit_classifier.py --webcam")
        sys.exit(0)

    process_image(sys.argv[1])
