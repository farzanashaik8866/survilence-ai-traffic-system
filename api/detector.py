"""
SURVILLENCE TRAFFIC — Accuracy-focused detection + tracking engine.

The project uses the bundled IISc UVH-26 YOLOv11-S vehicle model and a
lightweight ByteTrack-style tracker. This module intentionally avoids fake
results and adds temporal confirmation, class voting, robust direction/speed
smoothing, perspective-aware lane assignment, and configurable speed-gate
calibration.
"""
import os
import sys
import time
import json
import shutil
import subprocess
import math
import threading
from pathlib import Path
from collections import defaultdict, deque, Counter
from statistics import median

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PT = BASE_DIR / "traffic_ai_complete_improved" / "vehicle_traffic.pt"
MODEL_ONNX = BASE_DIR / "traffic_ai_complete_improved" / "vehicle_traffic.onnx"

UVH26_TO_APP = {
    "Hatchback": "car", "Sedan": "car", "SUV": "car", "MUV": "car",
    "Bus": "bus", "Truck": "truck", "Three-wheeler": "auto",
    "Two-wheeler": "motorcycle", "LCV": "truck", "Mini-bus": "bus",
    "tempo-traveller": None, "bicycle": None, "Van": "car",
    # "Others" is deliberately ignored instead of being silently relabeled.
    "Others": None,
}
APP_CLASSES = ("car", "motorcycle", "auto", "bus", "truck")
DISPLAY_LABELS = {
    "car": "CAR", "motorcycle": "MOTORCYCLE", "auto": "AUTO-RICKSHAW",
    "bus": "BUS", "truck": "TRUCK",
}
COLORS = {
    "car": (255, 200, 0), "motorcycle": (0, 220, 255), "auto": (0, 165, 255),
    "bus": (255, 80, 180), "truck": (180, 80, 255),
}

# IMPORTANT: this order must exactly match the class-id order embedded in the
# bundled UVH-26 ONNX/PT model. The previous portable build used an alphabetically
# rearranged list here, which made class id 7 (Two-wheeler) appear as Sedan/car.
UVH26_NAMES = [
    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler",
    "Two-wheeler", "LCV", "Mini-bus", "tempo-traveller", "bicycle", "Van", "Others"
]
UVH26_CLASS_COUNT = 14
assert len(UVH26_NAMES) == UVH26_CLASS_COUNT


def normalize_source_label(label: str) -> str:
    clean = str(label).strip().lower().replace("_", " ")
    aliases = {
        "hatchback": "Hatchback", "sedan": "Sedan", "suv": "SUV", "muv": "MUV",
        "bus": "Bus", "truck": "Truck", "three-wheeler": "Three-wheeler",
        "three wheeler": "Three-wheeler", "auto": "Three-wheeler",
        "autorickshaw": "Three-wheeler", "auto-rickshaw": "Three-wheeler",
        "two-wheeler": "Two-wheeler", "two wheeler": "Two-wheeler",
        "motorcycle": "Two-wheeler", "motorbike": "Two-wheeler", "bike": "Two-wheeler",
        "bicycle": "bicycle", "lcv": "LCV", "mini-bus": "Mini-bus", "mini bus": "Mini-bus",
        "van": "Van", "tempo-traveller": "tempo-traveller", "tempo traveller": "tempo-traveller",
        "others": "Others", "other": "Others",
    }
    return aliases.get(clean, str(label))


def app_label_from_source(label: str):
    return UVH26_TO_APP.get(normalize_source_label(label))


def source_label_from_class_id(class_id: int) -> str:
    """Return the exact UVH-26 label for an ONNX class id."""
    try:
        idx = int(class_id)
    except Exception:
        return "Others"
    if 0 <= idx < len(UVH26_NAMES):
        return UVH26_NAMES[idx]
    return "Others"


def traffic_level(active_count: int) -> str:
    if active_count <= 8:
        return "LOW"
    if active_count <= 20:
        return "MEDIUM"
    if active_count <= 35:
        return "HIGH"
    return "SEVERE"


def _safe_float(value, default):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _lane_boundaries(width, height, lane_count, top_width_ratio, bottom_width_ratio):
    """Return trapezoid lane divider endpoints for a perspective-aware road."""
    lane_count = max(2, int(lane_count))
    top_width_ratio = min(1.0, max(0.35, float(top_width_ratio)))
    bottom_width_ratio = min(1.0, max(top_width_ratio, float(bottom_width_ratio)))
    top_left = (1.0 - top_width_ratio) / 2.0
    bottom_left = (1.0 - bottom_width_ratio) / 2.0
    lines = []
    for idx in range(1, lane_count):
        frac = idx / lane_count
        top_x = (top_left + top_width_ratio * frac) * width
        bottom_x = (bottom_left + bottom_width_ratio * frac) * width
        lines.append(((int(top_x), int(height * 0.15)), (int(bottom_x), height - 1)))
    return lines


def _lane_for_point(cx, cy, width, height, lane_count, top_width_ratio, bottom_width_ratio):
    """Assign bottom-center point to a perspective trapezoid lane."""
    top_width_ratio = min(1.0, max(0.35, float(top_width_ratio)))
    bottom_width_ratio = min(1.0, max(top_width_ratio, float(bottom_width_ratio)))
    y0 = height * 0.15
    t = 0.0 if cy <= y0 else min(1.0, (cy - y0) / max(1.0, height - y0))
    left = ((1.0 - top_width_ratio) / 2.0) * (1.0 - t) + ((1.0 - bottom_width_ratio) / 2.0) * t
    usable = top_width_ratio * (1.0 - t) + bottom_width_ratio * t
    x_norm = cx / max(1.0, width)
    rel = (x_norm - left) / max(usable, 1e-6)
    lane = int(math.floor(rel * lane_count)) + 1
    return max(1, min(lane_count, lane))


def _median_abs_deviation(values):
    if not values:
        return 0.0
    m = median(values)
    return median([abs(x - m) for x in values])


class KalmanFilterBox:
    def __init__(self, bbox):
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        x = float(bbox[0]) + w / 2.0
        y = float(bbox[1]) + h / 2.0
        self.mean = np.array([x, y, w / h, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.covariance = np.diag([10.0, 10.0, 1.0, 10.0, 1000.0, 1000.0, 10.0, 1000.0])

    def predict(self):
        self.mean[0] += self.mean[4]
        self.mean[1] += self.mean[5]
        self.mean[2] = max(0.05, self.mean[2] + self.mean[6])
        self.mean[3] = max(2.0, self.mean[3] + self.mean[7])
        self.covariance += np.diag([1.0, 1.0, 0.01, 1.0, 1.0, 1.0, 0.01, 1.0])

    def update(self, bbox):
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        x = float(bbox[0]) + w / 2.0
        y = float(bbox[1]) + h / 2.0
        z = np.array([x, y, w / h, h], dtype=float)
        H = np.zeros((4, 8), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        # Slightly more conservative measurement noise reduces jitter.
        R = np.diag([1.5, 1.5, 0.02, 1.5])
        residual = z - (H @ self.mean)
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.mean += K @ residual
        self.covariance = (np.eye(8) - K @ H) @ self.covariance

    def to_xyxy(self):
        x, y, a, h = self.mean[:4]
        w = max(1.0, a * h)
        return np.array([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0])


def compute_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix


class SimpleByteTrack:
    """ByteTrack-style association with class voting and temporal confirmation."""
    def __init__(self, high_thresh=0.45, low_thresh=0.20, max_lost_frames=18, match_iou=0.22):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_lost_frames = max_lost_frames
        self.match_iou = match_iou
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _point_from_box(box):
        # Bottom center is more stable for road scenes than bbox center.
        return ((box[0] + box[2]) / 2.0, float(box[3]))

    def _new_track(self, det, frame_index):
        tid = self.next_id
        self.next_id += 1
        px, py = self._point_from_box(det["box"])
        self.tracks[tid] = {
            "kf": KalmanFilterBox(det["box"]),
            "box": det["box"],
            "class": det["class"],
            "class_votes": Counter({det["class"]: 1.5}),
            "conf_history": deque([float(det["score"])], maxlen=24),
            "lost": 0,
            "frames": 1,
            "history": deque([(px, py, frame_index)], maxlen=120),
        }
        return tid

    def _apply_match(self, tid, det, frame_index, weight=1.0):
        t = self.tracks[tid]
        t["kf"].update(det["box"])
        t["box"] = [float(v) for v in det["box"]]
        t["lost"] = 0
        t["frames"] += 1
        t["class_votes"][det["class"]] += max(0.1, weight * float(det["score"]))
        t["class"] = t["class_votes"].most_common(1)[0][0]
        t["conf_history"].append(float(det["score"]))
        px, py = self._point_from_box(det["box"])
        t["history"].append((px, py, frame_index))

    def update(self, detections, frame_index=None):
        for t in self.tracks.values():
            t["kf"].predict()
            t["box"] = t["kf"].to_xyxy().tolist()
            t["lost"] += 1

        high = [d for d in detections if d["score"] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]
        active_ids = [tid for tid, t in self.tracks.items() if t["lost"] <= self.max_lost_frames]
        matched_t = set()
        matched_d = set()

        def associate(track_ids, dets, iou_threshold, weight):
            if not track_ids or not dets:
                return
            ious = compute_iou_matrix([self.tracks[tid]["box"] for tid in track_ids], [d["box"] for d in dets])
            used_t, used_d = set(), set()
            while True:
                best = None
                for ti, tid in enumerate(track_ids):
                    if ti in used_t:
                        continue
                    for di, det in enumerate(dets):
                        if di in used_d:
                            continue
                        iou = float(ious[ti, di])
                        if iou < iou_threshold:
                            continue
                        class_bonus = 0.04 if det["class"] == self.tracks[tid]["class"] else 0.0
                        score = iou + class_bonus
                        if best is None or score > best[0]:
                            best = (score, ti, di)
                if best is None:
                    break
                _, ti, di = best
                tid = track_ids[ti]
                self._apply_match(tid, dets[di], frame_index, weight=weight)
                used_t.add(ti); used_d.add(di); matched_t.add(tid)
                if weight > 1.0:
                    matched_d.add(di)

        associate(active_ids, high, self.match_iou, 1.4)
        unmatched_active = [tid for tid in active_ids if tid not in matched_t]
        associate(unmatched_active, low, max(0.28, self.match_iou + 0.04), 0.8)

        for di, det in enumerate(high):
            if di not in matched_d:
                self._new_track(det, frame_index)

        dead = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost_frames]
        for tid in dead:
            del self.tracks[tid]

        results = []
        for tid, t in self.tracks.items():
            if t["lost"] == 0:
                results.append({
                    "track_id": tid,
                    "box": [float(v) for v in t["box"]],
                    "class": t["class"],
                    "conf": float(median(list(t["conf_history"])[-7:])),
                    "frames": int(t["frames"]),
                    "history": list(t["history"]),
                })
        return results


_cached_detector = None
_INFERENCE_LOCK = threading.RLock()


def get_yolo_detector():
    """Load the bundled ONNX model with OpenCV first for a portable local runtime.

    The project intentionally prefers the ONNX graph over the optional PyTorch
    checkpoint so a fresh Windows laptop does not need the very large torch /
    ultralytics dependency chain just to run traffic analysis. The PyTorch path
    remains as a fallback for environments that already have Ultralytics installed.
    """
    global _cached_detector
    if _cached_detector is not None:
        return _cached_detector

    if MODEL_ONNX.exists():
        try:
            net = cv2.dnn.readNetFromONNX(str(MODEL_ONNX))
            print(f"[Detector] Loaded portable ONNX UVH-26 with OpenCV DNN: {MODEL_ONNX}", flush=True)
            _cached_detector = ("opencv-onnx", net)
            return _cached_detector
        except Exception as exc:
            print(f"[Detector] OpenCV ONNX load skipped ({exc})", flush=True)

    if MODEL_PT.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(MODEL_PT))
            print(f"[Detector] Loaded optional PyTorch UVH-26: {MODEL_PT}", flush=True)
            _cached_detector = ("ultralytics", model)
            return _cached_detector
        except Exception as exc:
            print(f"[Detector] Optional Ultralytics load skipped ({exc})", flush=True)

    return None


def _letterbox(frame, out_w, out_h, pad_value=114):
    """Resize with preserved aspect ratio and return scale + padding."""
    h, w = frame.shape[:2]
    scale = min(float(out_w) / max(w, 1), float(out_h) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def run_inference_on_frame(detector, frame, conf_thresh=0.30, imgsz=640, nms_iou=0.45):
    engine_type, model = detector
    h, w = frame.shape[:2]
    min_box_area = max(16.0, 0.00018 * (w * h))
    detections = []

    if engine_type == "ultralytics":
        with _INFERENCE_LOCK:
            result = model(frame, conf=conf_thresh, imgsz=imgsz, iou=nms_iou, verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < min_box_area:
                    continue
                source = result.names.get(int(cls_id), str(cls_id))
                app_class = app_label_from_source(source)
                if not app_class:
                    continue
                detections.append({"box": [x1, y1, x2, y2], "score": float(conf), "class": app_class})
        return detections

    # Portable OpenCV-ONNX path. Handles YOLO outputs [1, 18, N] and [1, N, 18].
    if engine_type == "opencv-onnx":
        # The bundled UVH-26 ONNX graph is exported with a fixed 640x640 input.
        # Keep this exact shape for OpenCV DNN portability; frame_stride and max_dim
        # remain available as the local performance controls.
        onnx_size = 640
        input_img, ratio, pad_x, pad_y = _letterbox(frame, onnx_size, onnx_size)
        blob = cv2.dnn.blobFromImage(input_img, scalefactor=1 / 255.0, size=(onnx_size, onnx_size), swapRB=True, crop=False)
        with _INFERENCE_LOCK:
            model.setInput(blob)
            preds = np.asarray(model.forward())

        if preds.ndim == 3:
            preds = preds[0]
        if preds.ndim != 2:
            return detections
        if preds.shape[0] < preds.shape[1] and preds.shape[0] <= 32:
            preds = preds.T
        if preds.shape[1] < 5:
            return detections

        boxes_raw = preds[:, :4]
        scores_raw = preds[:, 4:]
        if scores_raw.size == 0:
            return detections
        # Some exports emit logits, while the bundled UVH-26 export emits sigmoid scores.
        if float(np.nanmax(scores_raw)) > 1.0001 or float(np.nanmin(scores_raw)) < 0.0:
            scores_raw = 1.0 / (1.0 + np.exp(-np.clip(scores_raw, -20, 20)))
        max_scores = np.max(scores_raw, axis=1)
        max_classes = np.argmax(scores_raw, axis=1)
        mask = np.isfinite(max_scores) & (max_scores >= conf_thresh)
        boxes_raw = boxes_raw[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = max_classes[mask]
        if len(boxes_raw) == 0:
            return detections

        nms_boxes, mapped, mapped_scores, mapped_classes = [], [], [], []
        for b, score, cls_idx in zip(boxes_raw, filtered_scores, filtered_classes):
            cx, cy, bw, bh = [float(x) for x in b]
            # Normalized exports use 0..1 coordinates; pixel-space exports use input pixels.
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
                cx *= onnx_size; cy *= onnx_size; bw *= onnx_size; bh *= onnx_size
            x1_in = cx - bw / 2.0
            y1_in = cy - bh / 2.0
            x2_in = cx + bw / 2.0
            y2_in = cy + bh / 2.0
            # Undo letterbox padding before mapping back to the source frame.
            x1 = (x1_in - pad_x) / max(ratio, 1e-9)
            y1 = (y1_in - pad_y) / max(ratio, 1e-9)
            x2 = (x2_in - pad_x) / max(ratio, 1e-9)
            y2 = (y2_in - pad_y) / max(ratio, 1e-9)
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(h - 1, int(round(y1))))
            x2 = max(0, min(w - 1, int(round(x2))))
            y2 = max(0, min(h - 1, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            nms_boxes.append([x1, y1, x2 - x1, y2 - y1])
            mapped.append([x1, y1, x2, y2])
            mapped_scores.append(float(score))
            mapped_classes.append(int(cls_idx))

        if not nms_boxes:
            return detections
        indices = cv2.dnn.NMSBoxes(nms_boxes, mapped_scores, conf_thresh, nms_iou)
        if len(indices) == 0:
            return detections
        for idx in np.asarray(indices).reshape(-1):
            idx = int(idx)
            x1, y1, x2, y2 = mapped[idx]
            area = (x2 - x1) * (y2 - y1)
            if area < min_box_area:
                continue
            cls_idx = mapped_classes[idx]
            source = source_label_from_class_id(cls_idx)
            app_class = app_label_from_source(source)
            if not app_class:
                continue
            detections.append({"box": [x1, y1, x2, y2], "score": float(mapped_scores[idx]), "class": app_class})
        return detections

    return detections


def _robust_motion(history, fps, window=10, min_motion_px=10.0, speed_window=5):
    if len(history) < 2:
        return "UNKNOWN", 0.0, 0.0
    pts = history[-max(2, window):]
    dy = pts[-1][1] - pts[0][1]
    total_abs_dy = 0.0
    dt_total = (pts[-1][2] - pts[0][2]) / max(fps, 1.0)
    seg_speeds = []
    for a, b in zip(pts[:-1], pts[1:]):
        frame_dt = (b[2] - a[2]) / max(fps, 1.0)
        if frame_dt <= 0:
            continue
        seg_dy = b[1] - a[1]
        total_abs_dy += abs(seg_dy)
        seg_dist = math.hypot(b[0] - a[0], b[1] - a[1])
        seg_speeds.append(seg_dist / frame_dt)
    if not seg_speeds or dt_total <= 0:
        return "UNKNOWN", 0.0, 0.0
    # Reject isolated tracking jumps using a median/MAD filter.
    m = median(seg_speeds)
    mad = _median_abs_deviation(seg_speeds)
    if mad > 0:
        filtered = [s for s in seg_speeds if abs(s - m) <= max(3.0 * mad, 0.45 * m)]
        if filtered:
            seg_speeds = filtered
    speed_px_s = float(median(seg_speeds[-max(2, speed_window):]))
    net_speed_px_s = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) / dt_total
    # Never let a single jitter sequence create a giant estimate.
    speed_px_s = min(speed_px_s, max(net_speed_px_s * 1.8, 4.0))

    direction = "UNKNOWN"
    if abs(dy) >= min_motion_px and total_abs_dy >= abs(dy):
        directional_consistency = abs(dy) / max(total_abs_dy, 1e-6)
        if directional_consistency >= 0.58:
            direction = "DOWN" if dy > 0 else "UP"
    return direction, speed_px_s, net_speed_px_s


def _crossing_time(history_a, history_b, target_y):
    y0, f0 = history_a[1], history_a[2]
    y1, f1 = history_b[1], history_b[2]
    if f1 <= f0 or y1 == y0:
        return None
    if (y0 - target_y) * (y1 - target_y) > 0:
        return None
    ratio = (target_y - y0) / (y1 - y0)
    return float(f0 + ratio * (f1 - f0))


def _gate_speed_kmh(history, fps, reference_distance_m, gate_top_y, gate_bottom_y):
    """Compute a physical speed from two virtual gates.

    Each gate crossing is detected over the complete recent track history, not just
    the final 3-4 points. That prevents missed violations when gates are separated
    by many frames.
    """
    if reference_distance_m <= 0 or len(history) < 2:
        return None, {}
    crossings = {}
    for a, b in zip(history[:-1], history[1:]):
        if 'gate_a' not in crossings:
            t = _crossing_time(a, b, gate_top_y)
            if t is not None:
                crossings['gate_a'] = t
        if 'gate_b' not in crossings:
            t = _crossing_time(a, b, gate_bottom_y)
            if t is not None:
                crossings['gate_b'] = t
        if 'gate_a' in crossings and 'gate_b' in crossings:
            break
    if 'gate_a' not in crossings or 'gate_b' not in crossings:
        return None, crossings
    dt = abs(crossings['gate_b'] - crossings['gate_a']) / max(fps, 1.0)
    if dt < 0.05 or dt > 30.0:
        return None, crossings
    return float((reference_distance_m / dt) * 3.6), crossings


def _speed_violation_confirmed(readings, threshold_kmh, min_hits=3, window=5):
    """Require sustained evidence before marking a speed violation.

    This works for both estimated and gate-calibrated speeds and intentionally
    ignores isolated one-frame spikes.
    """
    vals = [float(v) for v in list(readings)[-window:] if math.isfinite(float(v)) and float(v) > 0]
    if len(vals) < min_hits:
        return False
    hits = sum(1 for v in vals if v >= threshold_kmh)
    return hits >= min_hits


def process_video_analysis(
    input_path: str,
    output_video_path: str,
    output_report_path: str,
    session_id: str,
    progress_callback=None,
    stride: int = 1,
    max_dim: int = 1280,
    settings: dict | None = None,
    cancel_check=None,
) -> dict:
    settings = settings or {}
    expected_direction = str(settings.get("expected_direction", "AUTO")).upper()
    if expected_direction not in ("UP", "DOWN", "BOTH", "AUTO"):
        expected_direction = "AUTO"
    conf_thresh = min(0.70, max(0.10, _safe_float(settings.get("confidence", settings.get("detect_confidence", 0.30)), 0.30)))
    imgsz = max(416, min(960, _safe_int(settings.get("image_size", settings.get("imgsz", 640)), 640)))
    stride = max(1, min(4, _safe_int(settings.get("frame_stride", stride), stride)))
    nms_iou = min(0.70, max(0.30, _safe_float(settings.get("nms_iou", 0.45), 0.45)))
    lane_count = max(2, min(8, _safe_int(settings.get("lane_count", 3), 3)))
    speed_limit_kmh = max(5.0, min(180.0, _safe_float(settings.get("speed_limit_kmh", 40), 40)))
    meters_per_pixel = max(0.0001, min(5.0, _safe_float(settings.get("meters_per_pixel", 0.05), 0.05)))
    stopped_seconds = max(2.0, min(60.0, _safe_float(settings.get("stopped_seconds", 5), 5)))
    speed_margin_kmh = max(0.0, min(15.0, _safe_float(settings.get("speed_margin_kmh", 3), 3)))
    confirmation_frames = max(2, min(8, _safe_int(settings.get("confirmation_frames", 3), 3)))
    direction_window = max(5, min(20, _safe_int(settings.get("direction_window", 10), 10)))
    direction_min_pixels = max(4.0, min(60.0, _safe_float(settings.get("direction_min_pixels", 10), 10)))
    speed_smoothing_window = max(3, min(10, _safe_int(settings.get("speed_smoothing_window", 5), 5)))
    lane_top_width_ratio = min(1.0, max(0.35, _safe_float(settings.get("lane_top_width_ratio", 0.60), 0.60)))
    lane_bottom_width_ratio = min(1.0, max(lane_top_width_ratio, _safe_float(settings.get("lane_bottom_width_ratio", 1.0), 1.0)))
    reference_distance_m = max(0.0, min(1000.0, _safe_float(settings.get("speed_reference_distance_m", 0.0), 0.0)))
    gate_top = min(0.80, max(0.10, _safe_float(settings.get("speed_gate_top", 0.35), 0.35)))
    gate_bottom = min(0.95, max(gate_top + 0.05, _safe_float(settings.get("speed_gate_bottom", 0.70), 0.70)))
    speed_mode = str(settings.get("speed_mode", "AUTO")).upper()
    if speed_mode not in ("ESTIMATED", "GATES", "AUTO"):
        speed_mode = "AUTO"
    speed_confirm_window = max(3, min(8, _safe_int(settings.get("speed_confirmation_window", 5), 5)))
    speed_confirm_hits = max(2, min(speed_confirm_window, _safe_int(settings.get("speed_confirmation_hits", 3), 3)))

    start_time = time.time()
    detector = get_yolo_detector()
    if not detector:
        raise RuntimeError("UVH-26 YOLO model could not be loaded. Install the project's requirements and retry.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open uploaded video file: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if orig_w <= 0 or orig_h <= 0:
        cap.release(); raise ValueError("Video dimensions could not be read.")

    scale = min(1.0, float(max_dim) / max(orig_w, orig_h))
    width = max(2, int(orig_w * scale) // 2 * 2)
    height = max(2, int(orig_h * scale) // 2 * 2)

    tracker = SimpleByteTrack(high_thresh=max(conf_thresh + 0.10, 0.45), low_thresh=max(0.18, conf_thresh - 0.08))
    track_frames = Counter()
    started_tracks = set()
    confirmed_track_ids = set()
    unique_vehicle_classes = {}
    vehicles_detail = {}
    vehicle_lane_sets = defaultdict(set)
    lane_speed_samples = defaultdict(list)
    per_frame_draw_items = []
    per_frame_active_counts = []
    per_frame_lane_counts = []
    per_frame_centers = []
    incidents = []
    incidents_seen = set()
    wrong_way_tracks = set(); speed_violation_tracks = set(); stopped_tracks = set()
    alerts_by_track = defaultdict(list)
    direction_votes = defaultdict(Counter)
    speed_samples_by_track = defaultdict(lambda: deque(maxlen=15))
    speed_kmh_samples_by_track = defaultdict(lambda: deque(maxlen=15))
    gate_speed_by_track = {}
    direction_stable_first_frame = {}
    auto_wrong_way_activation = {}
    auto_expected_by_lane = {}
    auto_detected_direction = "UNKNOWN"
    stationarity_started = {}
    gate_crossings = defaultdict(dict)
    active_history = deque(maxlen=max(20, int(fps * 20)))
    severe_alert_last_frame = -10**9

    def add_incident(event_type, severity, frame_number, message, tid=None, vehicle_type=None, lane=None):
        key = (tid, event_type) if tid is not None else (event_type, int(frame_number / max(fps * 3, 1)))
        if key in incidents_seen:
            return
        incidents_seen.add(key)
        incidents.append({
            "id": f"inc_{session_id[:8]}_{len(incidents)+1:04d}",
            "type": event_type, "severity": severity, "track_id": int(tid) if tid is not None else None,
            "vehicle_type": vehicle_type, "lane": int(lane) if lane is not None else None,
            "frame": int(frame_number), "time_seconds": round(frame_number / max(fps, 1.0), 2),
            "message": message,
        })
        if tid is not None:
            alerts_by_track[tid].append(event_type)

    if progress_callback:
        progress_callback(8, "Opening video and calibrating the AI analysis pipeline...")

    frame_number = 0
    last_draw_items = []
    last_lane_counts = {}
    while True:
        if cancel_check and cancel_check():
            cap.release(); raise RuntimeError("__ANALYSIS_CANCELLED__")
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        if scale < 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        infer = (frame_number == 1) or (frame_number % stride == 0)
        if infer:
            detections = run_inference_on_frame(detector, frame, conf_thresh=conf_thresh, imgsz=imgsz, nms_iou=nms_iou)
            active_tracks = tracker.update(detections, frame_index=frame_number)
            current_active = len(active_tracks)
            frame_lane_counts = defaultdict(int)
            frame_centers = []
            draw_items = []

            for t in active_tracks:
                tid = t["track_id"]
                started_tracks.add(tid); track_frames[tid] += 1
                app_class = t["class"]; unique_vehicle_classes[tid] = app_class
                box = [int(round(v)) for v in t["box"]]
                cx = int((box[0] + box[2]) / 2); cy_bottom = int(box[3])
                lane = _lane_for_point(cx, cy_bottom, width, height, lane_count, lane_top_width_ratio, lane_bottom_width_ratio)
                lane_key = f"Lane {lane}"; frame_lane_counts[lane_key] += 1; vehicle_lane_sets[lane_key].add(tid)
                frame_centers.append((cx, cy_bottom, tid))

                hist = t.get("history", [])
                direction, speed_px_s, net_speed_px_s = _robust_motion(hist, fps, direction_window, direction_min_pixels, speed_smoothing_window)
                if direction in ("UP", "DOWN"):
                    direction_votes[tid][direction] += 1
                speed_samples_by_track[tid].append(speed_px_s)
                smoothed_px_s = float(median(list(speed_samples_by_track[tid])[-speed_smoothing_window:])) if speed_samples_by_track[tid] else 0.0
                estimated_kmh = smoothed_px_s * meters_per_pixel * 3.6
                speed_kmh = float(estimated_kmh)
                calibrated_kmh = None

                # Real-distance speed gates: scan the full track history so a fast
                # vehicle cannot pass the gates between sparse inference points.
                if reference_distance_m > 0 and len(hist) >= 2:
                    calibrated_kmh, crossings = _gate_speed_kmh(
                        hist, fps, reference_distance_m, height * gate_top, height * gate_bottom
                    )
                    if crossings:
                        gate_crossings[tid].update(crossings)
                    if calibrated_kmh is not None:
                        gate_speed_by_track[tid] = float(calibrated_kmh)

                # AUTO chooses gate speed when a valid gate measurement exists;
                # otherwise estimated motion remains fully usable. ESTIMATED always
                # uses the calibrated meters-per-pixel estimate and GATES prefers only
                # a real gate measurement. The default is ESTIMATED so the speed
                # violation feature works immediately on a fresh installation.
                if speed_mode == "GATES":
                    speed_kmh = float(calibrated_kmh) if calibrated_kmh is not None else float(estimated_kmh)
                    speed_source = "GATE" if calibrated_kmh is not None else "ESTIMATE_FALLBACK"
                elif speed_mode == "AUTO":
                    speed_kmh = float(calibrated_kmh) if calibrated_kmh is not None else float(estimated_kmh)
                    speed_source = "GATE" if calibrated_kmh is not None else "ESTIMATE"
                else:
                    speed_kmh = float(estimated_kmh)
                    speed_source = "ESTIMATE"
                if speed_kmh > 0:
                    lane_speed_samples[lane_key].append(speed_kmh)
                    speed_kmh_samples_by_track[tid].append(speed_kmh)

                direction_consistent = direction_votes[tid].most_common(1)[0][0] if direction_votes[tid] else direction
                vote_total = sum(direction_votes[tid].values()) or 1
                top_votes = direction_votes[tid].most_common(1)[0][1] if direction_votes[tid] else 0
                strong_direction = direction_consistent if top_votes / vote_total >= 0.65 and top_votes >= 2 else "UNKNOWN"
                # AUTO wrong-way evaluation is intentionally deferred until the complete
                # scene is observed. A single vehicle cannot establish its own legal
                # direction, and mixed-flow lanes are not treated as violations.
                wrong_way = expected_direction in ("UP", "DOWN") and strong_direction in ("UP", "DOWN") and strong_direction != expected_direction

                violation_threshold = speed_limit_kmh + speed_margin_kmh
                # Speed violations are operationally useful even without gate
                # calibration. In that mode the value is a camera-calibrated estimate,
                # so require sustained threshold evidence instead of a one-frame spike.
                speed_violation_confirmed = False
                if track_frames[tid] >= max(confirmation_frames + 1, speed_confirm_hits):
                    speed_violation_confirmed = _speed_violation_confirmed(
                        speed_kmh_samples_by_track[tid], violation_threshold,
                        min_hits=speed_confirm_hits, window=speed_confirm_window
                    )
                if speed_violation_confirmed:
                    speed_violation_tracks.add(tid)
                    add_incident(
                        "SPEED_VIOLATION", "HIGH", frame_number,
                        f"Vehicle #{tid} measured at {speed_kmh:.1f} km/h; configured limit {speed_limit_kmh:.1f} km/h",
                        tid, app_class, lane
                    )

                wrong_way_confirmed = bool(expected_direction in ("UP", "DOWN") and wrong_way and direction_votes[tid][strong_direction] >= 3)
                if wrong_way_confirmed:
                    wrong_way_tracks.add(tid)
                    add_incident("WRONG_WAY", "HIGH", frame_number, f"Vehicle #{tid} moving {strong_direction}; expected {expected_direction}", tid, app_class, lane)

                # Stationary incident: measure actual elapsed video time rather than frame count alone.
                if speed_px_s <= 2.4 and len(hist) >= 3:
                    stationarity_started.setdefault(tid, frame_number)
                    stationary_for = (frame_number - stationarity_started[tid]) / max(fps, 1.0)
                    had_motion = any(float(x) > 5.0 for x in list(speed_samples_by_track[tid])[:-2])
                    if stationary_for >= stopped_seconds and had_motion:
                        stopped_tracks.add(tid)
                        add_incident("STOPPED_VEHICLE", "MEDIUM", frame_number, f"Vehicle #{tid} has remained nearly stationary for {stopped_seconds:.0f}+ seconds", tid, app_class, lane)
                else:
                    stationarity_started.pop(tid, None)

                if track_frames[tid] >= confirmation_frames:
                    confirmed_track_ids.add(tid)

                vehicles_detail[tid] = {
                    "id": int(tid), "track_id": int(tid), "class": app_class, "type": app_class,
                    "conf": round(float(t["conf"]), 3), "direction": strong_direction, "lane": lane,
                    "speed_px_s": round(smoothed_px_s, 2), "speed_kmh": round(float(speed_kmh), 2),
                    "speed_violation": bool(tid in speed_violation_tracks),
                    "wrong_way": bool(tid in wrong_way_tracks), "status": "INCIDENT" if alerts_by_track.get(tid) else "TRACKED",
                    "incidents": list(alerts_by_track.get(tid, [])), "frames_tracked": int(track_frames[tid]),
                    "speed_estimated": speed_source in ("ESTIMATE", "ESTIMATE_FALLBACK"),
                    "speed_calibrated": speed_source == "GATE",
                    "speed_source": speed_source,
                    "calibration_mode": "reference-gates" if speed_source == "GATE" else "meters-per-pixel-estimate",
                }
                color = (40, 80, 255) if tid in wrong_way_tracks else ((0, 80, 255) if tid in speed_violation_tracks else COLORS.get(app_class, (200, 200, 200)))
                extra = f" L{lane}"
                if speed_kmh > 0:
                    extra += f" {speed_kmh:.0f}km/h"
                if tid in wrong_way_tracks: extra += " WRONG WAY"
                elif tid in speed_violation_tracks: extra += " SPEED"
                label = f"ID:{tid} {DISPLAY_LABELS.get(app_class, app_class)} {t['conf']:.0%}{extra}"
                draw_items.append((box[0], box[1], box[2], box[3], cx, cy_bottom, color, label, tid, strong_direction))

            # Trend-aware congestion spike alert.
            active_history.append((frame_number, current_active))
            recent = [c for _, c in list(active_history)[-min(len(active_history), int(fps * 5)):]]
            previous = [c for _, c in list(active_history)[-min(len(active_history), int(fps * 15)):-min(len(active_history), int(fps * 5))]]
            if len(previous) >= 3 and len(recent) >= 3:
                old_mean = float(np.mean(previous)); new_mean = float(np.mean(recent))
                if new_mean >= 15 and new_mean > old_mean * 1.45 and frame_number - severe_alert_last_frame > int(fps * 8):
                    severe_alert_last_frame = frame_number
                    add_incident("CONGESTION_SPIKE", "HIGH", frame_number, f"Active traffic increased sharply from ~{old_mean:.0f} to ~{new_mean:.0f} vehicles")
            if current_active > 35 and frame_number - severe_alert_last_frame > int(fps * 10):
                severe_alert_last_frame = frame_number
                add_incident("HIGH_CONGESTION", "HIGH", frame_number, f"High traffic density detected with {current_active} active vehicles")

            # Persist unconfirmed tracks in frames for later filtering.
            last_draw_items = draw_items
            last_lane_counts = dict(frame_lane_counts)
            per_frame_draw_items.append(draw_items)
            per_frame_lane_counts.append(dict(frame_lane_counts))
            per_frame_centers.append(frame_centers)
            per_frame_active_counts.append(current_active)
        else:
            per_frame_draw_items.append(last_draw_items)
            per_frame_lane_counts.append(last_lane_counts)
            per_frame_centers.append(per_frame_centers[-1] if per_frame_centers else [])
            per_frame_active_counts.append(per_frame_active_counts[-1] if per_frame_active_counts else 0)

        if total_frames and frame_number % max(10, stride * 10) == 0 and progress_callback:
            progress_callback(10 + int((frame_number / total_frames) * 60), f"Detecting + tracking ({frame_number}/{total_frames})...")

    cap.release()

    confirmed_track_ids = {tid for tid in confirmed_track_ids if track_frames[tid] >= confirmation_frames and tid in vehicles_detail}

    # ────────────────────────────────────────────────────────────────────────
    # Wrong-way logic
    # ────────────────────────────────────────────────────────────────────────
    # The previous implementation used DOWN as the default camera direction.
    # That is unsafe because many road cameras view traffic moving UP in image
    # coordinates (away from the camera). In AUTO mode we learn the dominant
    # direction independently for each lane, but only when at least two
    # vehicles provide consistent evidence. This prevents a legitimate vehicle
    # from being labelled wrong-way simply because the camera is facing the
    # opposite direction.
    stable_direction_by_track = {}
    stable_direction_ratio = {}
    for tid in confirmed_track_ids:
        votes = direction_votes.get(tid, Counter())
        vote_total = sum(votes.values())
        if not vote_total:
            stable_direction_by_track[tid] = "UNKNOWN"
            stable_direction_ratio[tid] = 0.0
            continue
        top_dir, top_votes = votes.most_common(1)[0]
        ratio = top_votes / max(vote_total, 1)
        if top_dir in ("UP", "DOWN") and top_votes >= 5 and ratio >= 0.72:
            stable_direction_by_track[tid] = top_dir
            stable_direction_ratio[tid] = ratio
        else:
            stable_direction_by_track[tid] = "UNKNOWN"
            stable_direction_ratio[tid] = ratio

    if expected_direction == "AUTO":
        lane_direction_votes = defaultdict(list)
        for tid in confirmed_track_ids:
            d = stable_direction_by_track.get(tid, "UNKNOWN")
            lane = (vehicles_detail.get(tid) or {}).get("lane")
            if d in ("UP", "DOWN") and lane is not None:
                lane_direction_votes[int(lane)].append(d)

        for lane, dirs in lane_direction_votes.items():
            # AUTO is deliberately conservative: at least three confirmed vehicles
            # with >=75% directional agreement are required before the lane gets a
            # learned legal flow. This avoids false alerts from a short track, camera
            # jitter, or a vehicle changing lanes near the lane boundary.
            if len(dirs) < 3:
                continue
            lane_counter = Counter(dirs)
            lane_dir, lane_n = lane_counter.most_common(1)[0]
            lane_ratio = lane_n / len(dirs)
            if lane_ratio >= 0.75:
                auto_expected_by_lane[int(lane)] = lane_dir

        global_dirs = [d for d in stable_direction_by_track.values() if d in ("UP", "DOWN")]
        if len(global_dirs) >= 2:
            global_counter = Counter(global_dirs)
            auto_detected_direction, global_n = global_counter.most_common(1)[0]
            global_ratio = global_n / len(global_dirs)
            if global_ratio < 0.85:
                auto_detected_direction = "MIXED"
        elif len(global_dirs) == 1:
            auto_detected_direction = global_dirs[0]

        for tid in confirmed_track_ids:
            direction = stable_direction_by_track.get(tid, "UNKNOWN")
            lane = int((vehicles_detail.get(tid) or {}).get("lane") or 0)
            lane_expected = auto_expected_by_lane.get(lane)
            # Never fall back to a global direction when the scene contains
            # credible mixed flow. That avoids flagging an opposing lane in a
            # two-way road.
            expected_for_track = lane_expected
            if expected_for_track and direction in ("UP", "DOWN") and direction != expected_for_track and stable_direction_ratio.get(tid, 0.0) >= 0.80:
                wrong_way_tracks.add(tid)
                # Use the most recent strongly evidenced history point as the
                # activation frame; the event is not emitted until after all
                # tracks are understood, so the final decision is scene-aware.
                track_history = (tracker.tracks.get(tid) or {}).get("history") or []
                activation_frame = int(direction_stable_first_frame.get(tid, track_history[-1][2] if track_history else frame_number))
                auto_wrong_way_activation[tid] = activation_frame
                add_incident(
                    "WRONG_WAY", "HIGH", activation_frame,
                    f"Vehicle #{tid} moving {direction}; lane flow is {expected_for_track}",
                    tid, unique_vehicle_classes.get(tid), lane or None
                )

    # Keep the authoritative counts after AUTO direction decisions.
    target_counts = {c: sum(1 for tid in confirmed_track_ids if unique_vehicle_classes.get(tid) == c) for c in APP_CLASSES}
    authoritative_total = len(confirmed_track_ids)

    final_vehicles_detail = []
    for display_id, tid in enumerate(sorted(confirmed_track_ids), start=1):
        v = dict(vehicles_detail[tid])
        if expected_direction == "AUTO":
            v["direction"] = stable_direction_by_track.get(tid, v.get("direction", "UNKNOWN"))
        v["id"] = display_id
        v["wrong_way"] = tid in wrong_way_tracks
        v["speed_violation"] = tid in speed_violation_tracks
        v["incidents"] = list(dict.fromkeys(alerts_by_track.get(tid, [])))
        if v["wrong_way"] or v["speed_violation"] or v["incidents"]:
            v["status"] = "INCIDENT"
        final_vehicles_detail.append(v)

    incidents = [e for e in incidents if e.get("track_id") is None or e.get("track_id") in confirmed_track_ids]
    wrong_way_count = sum(1 for v in final_vehicles_detail if v.get("wrong_way"))
    speed_violation_count = sum(1 for v in final_vehicles_detail if v.get("speed_violation"))
    stopped_count = sum(1 for v in final_vehicles_detail if "STOPPED_VEHICLE" in (v.get("incidents") or []))

    if progress_callback:
        progress_callback(75, f"Tracking complete — {authoritative_total} confirmed vehicles. Rendering intelligence overlay...")

    # Render pass.
    cap2 = cv2.VideoCapture(input_path)
    raw_output = str(Path(output_video_path).with_name(f"raw_{Path(output_video_path).name}"))
    writer = cv2.VideoWriter(raw_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(raw_output, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    lane_lines = _lane_boundaries(width, height, lane_count, lane_top_width_ratio, lane_bottom_width_ratio)
    frame_idx = 0
    while True:
        if cancel_check and cancel_check():
            cap2.release(); writer.release(); Path(raw_output).unlink(missing_ok=True); raise RuntimeError("__ANALYSIS_CANCELLED__")
        ok, frame = cap2.read()
        if not ok or frame_idx >= len(per_frame_draw_items):
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for idx, (top, bottom) in enumerate(lane_lines, start=1):
            cv2.line(frame, top, bottom, (90, 100, 115), 1)
            cv2.putText(frame, f"L{idx}", (top[0] + 5, top[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 190, 205), 1)
        if lane_count:
            cv2.putText(frame, f"L{lane_count}", (width - 42, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 190, 205), 1)
        for item in per_frame_draw_items[frame_idx]:
            x1, y1, x2, y2, cx, cy, color, label, tid, det_dir = item
            # AUTO wrong-way decisions are made after the full scene is observed.
            # Apply the final decision during rendering so the output video still
            # displays the alert consistently even though it was determined late.
            display_color = color
            display_label = label
            activation = auto_wrong_way_activation.get(tid)
            wrong_active = tid in wrong_way_tracks and (expected_direction != "AUTO" or activation is None or frame_idx + 1 >= activation)
            if wrong_active:
                display_color = (40, 80, 255)
                if "WRONG WAY" not in display_label:
                    display_label += " WRONG WAY"
            elif tid in speed_violation_tracks:
                display_color = (0, 80, 255)
                if "SPEED" not in display_label:
                    display_label += " SPEED"
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(frame, display_label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, display_color, 2)
            cv2.circle(frame, (cx, cy), 3, display_color, -1)
            tr = tracker.tracks.get(tid)
            if tr and len(tr.get("history", [])) >= 2:
                a, b = tr["history"][-2], tr["history"][-1]
                cv2.arrowedLine(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), display_color, 2, tipLength=0.25)
        cv2.rectangle(frame, (8, 8), (445, 325), (15, 20, 28), -1)
        cv2.rectangle(frame, (8, 8), (445, 325), (40, 55, 75), 1)
        current_active = per_frame_active_counts[frame_idx] if per_frame_active_counts else 0
        level = traffic_level(current_active)
        overlay = [
            f"SESSION: {session_id[:18]}", f"TOTAL UNIQUE: {authoritative_total}", f"ACTIVE NOW: {current_active} ({level})",
            f"CAR: {target_counts['car']}  MOTO: {target_counts['motorcycle']}", f"AUTO: {target_counts['auto']}  BUS: {target_counts['bus']}",
            f"TRUCK: {target_counts['truck']}", f"WRONG WAY: {wrong_way_count}", f"SPEED VIOLATIONS: {speed_violation_count}",
            f"FLOW: {auto_detected_direction if expected_direction == 'AUTO' else expected_direction}",
            f"INCIDENTS: {len(incidents)}", f"SPEED MODE: {('GATES' if speed_mode == 'GATES' else 'AUTO-GATE' if speed_mode == 'AUTO' and reference_distance_m > 0 else 'ESTIMATED')}",
        ]
        y = 30
        for line in overlay:
            color = (240, 245, 255)
            if "TOTAL UNIQUE" in line: color = (0, 255, 200)
            elif "ACTIVE NOW" in line: color = (100, 200, 255)
            elif "WRONG WAY" in line and wrong_way_count: color = (80, 100, 255)
            elif "SPEED VIOLATIONS" in line and speed_violation_count: color = (0, 160, 255)
            cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2); y += 28
        # Show the latest event briefly.
        frame_events = [e for e in incidents if e.get("frame", 0) <= frame_idx + 1 and e.get("frame", 0) > max(0, frame_idx + 1 - int(fps * 2))]
        if frame_events:
            msg = frame_events[-1]["type"]
            cv2.rectangle(frame, (max(10, width - 385), 10), (width - 10, 48), (45, 18, 22), -1)
            cv2.putText(frame, f"ALERT: {msg}", (max(18, width - 375), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 100, 255), 2)
        writer.write(frame); frame_idx += 1
    cap2.release(); writer.release()

    if progress_callback:
        progress_callback(90, "Encoding the annotated video for browser playback...")
    # Always try to produce browser-compatible H.264 MP4. Some Windows
    # installations do not have a system ffmpeg, which previously caused the
    # raw OpenCV mp4v file to be renamed to .mp4. That file may show as 0:00
    # or fail to play in Chrome/Edge on another laptop.
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

    if ffmpeg_bin and Path(raw_output).exists():
        temp_h264 = str(Path(output_video_path).with_name(f"h264_{Path(output_video_path).name}"))
        cmd = [ffmpeg_bin, "-y", "-i", raw_output, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", temp_h264]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and Path(temp_h264).exists() and Path(temp_h264).stat().st_size > 0:
                shutil.move(temp_h264, output_video_path)
            else:
                shutil.move(raw_output, output_video_path)
        except Exception:
            shutil.move(raw_output, output_video_path)
        Path(raw_output).unlink(missing_ok=True)
    elif Path(raw_output).exists():
        shutil.move(raw_output, output_video_path)

    elapsed = max(time.time() - start_time, 0.01)
    avg_active = float(np.mean(per_frame_active_counts)) if per_frame_active_counts else 0.0
    peak_active = int(max(per_frame_active_counts)) if per_frame_active_counts else 0
    direction_counts = Counter(v["direction"] for v in final_vehicles_detail if v.get("direction") in ("UP", "DOWN"))
    speed_values = [float(v.get("speed_kmh", 0)) for v in final_vehicles_detail if float(v.get("speed_kmh", 0)) > 0]
    avg_speed_kmh = round(float(np.median(speed_values)), 2) if speed_values else 0.0
    speed_values_px = [float(v.get("speed_px_s", 0)) for v in final_vehicles_detail if float(v.get("speed_px_s", 0)) > 0]
    avg_speed_px_s = round(float(np.median(speed_values_px)), 2) if speed_values_px else 0.0

    lane_stats = {}
    for lane in range(1, lane_count + 1):
        key = f"Lane {lane}"; samples = lane_speed_samples.get(key, [])
        counts = [d.get(key, 0) for d in per_frame_lane_counts]
        lane_stats[key] = {
            "unique_vehicles": len(vehicle_lane_sets.get(key, set()) & confirmed_track_ids),
            "average_active": round(float(np.mean(counts)) if counts else 0.0, 2),
            "peak_active": int(max(counts, default=0)),
            "average_speed_kmh": round(float(np.median(samples)), 2) if samples else 0.0,
            "speed_samples": len(samples),
        }

    # Spatial occupancy heatmap (6 columns x 4 rows) from real tracked bottom-center positions.
    heat_rows, heat_cols = 4, 6
    heat_grid = [[0 for _ in range(heat_cols)] for _ in range(heat_rows)]
    for points in per_frame_centers[::max(1, stride)]:
        for cx, cy, tid in points:
            if tid not in confirmed_track_ids:
                continue
            gx = max(0, min(heat_cols - 1, int((cx / max(width, 1)) * heat_cols)))
            gy = max(0, min(heat_rows - 1, int((cy / max(height, 1)) * heat_rows)))
            heat_grid[gy][gx] += 1
    max_heat = max((max(row) for row in heat_grid), default=0)
    hotspots = []
    for r, row in enumerate(heat_grid):
        for c, value in enumerate(row):
            if value >= max(3, max_heat * 0.60):
                hotspots.append({"row": r + 1, "column": c + 1, "intensity": int(value)})

    started_count = len(started_tracks)
    confirmed_ratio = (authoritative_total / started_count) if started_count else 0.0
    stable_count = sum(1 for tid in confirmed_track_ids if track_frames[tid] >= max(confirmation_frames + 2, 5))
    stable_ratio = (stable_count / authoritative_total) if authoritative_total else 0.0
    conf_values = [float(v["conf"]) for v in final_vehicles_detail]
    avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
    quality_score = round(100 * (0.50 * min(avg_conf / 0.65, 1.0) + 0.30 * confirmed_ratio + 0.20 * stable_ratio))
    quality_level = "HIGH" if quality_score >= 78 else ("MEDIUM" if quality_score >= 55 else "LOW")

    bottleneck_lane = None
    if lane_stats:
        bottleneck_lane = max(lane_stats.items(), key=lambda kv: (kv[1]["average_active"], kv[1]["peak_active"]))[0]
    recommendations = []
    if reference_distance_m <= 0:
        recommendations.append("Speed violations use the configured meters-per-pixel estimate; two known-distance gates can be enabled for a more stable physical-speed measurement.")
    if wrong_way_count:
        recommendations.append(f"Review {wrong_way_count} confirmed wrong-way track(s) around their evidence frames.")
    if speed_violation_count:
        recommendations.append(f"Review {speed_violation_count} speed-violation track(s); the detector uses temporal confirmation to reduce one-frame false alerts.")
    if bottleneck_lane:
        recommendations.append(f"{bottleneck_lane} carried the highest sustained tracked load in this session.")
    if not recommendations:
        recommendations.append("No high-priority safety signal crossed the configured thresholds in this session.")

    report = {
        "session_id": session_id, "video": Path(input_path).name,
        "model": "IISc UVH-26 YOLOv11-S", "model_source": "Bundled UVH-26 YOLOv11-S",
        "frames_processed": frame_number, "processing_time_seconds": round(elapsed, 2),
        "analysis_duration": round(elapsed, 2), "video_duration": round(float(total_frames / fps), 2) if fps > 0 and total_frames else 0.0,
        "processing_fps": round(frame_number / elapsed, 2),
        "total_unique": authoritative_total, "total_vehicles": authoritative_total, "total_count": authoritative_total,
        "cars": int(target_counts["car"]), "motorcycles": int(target_counts["motorcycle"]), "auto_rickshaws": int(target_counts["auto"]),
        "buses": int(target_counts["bus"]), "trucks": int(target_counts["truck"]),
        "active_now": int(per_frame_active_counts[-1]) if per_frame_active_counts else 0,
        "vehicle_counts": {"car": int(target_counts["car"]), "motorcycle": int(target_counts["motorcycle"]), "auto": int(target_counts["auto"]), "auto_rickshaw": int(target_counts["auto"]), "bus": int(target_counts["bus"]), "truck": int(target_counts["truck"])},
        "vehicles": {"total": authoritative_total, "car": int(target_counts["car"]), "motorcycle": int(target_counts["motorcycle"]), "auto": int(target_counts["auto"]), "auto_rickshaw": int(target_counts["auto"]), "bus": int(target_counts["bus"]), "truck": int(target_counts["truck"])},
        "direction": {"up": int(direction_counts["UP"]), "down": int(direction_counts["DOWN"]), "expected": expected_direction, "auto_detected": auto_detected_direction, "auto_lane_expected": {str(k): v for k, v in sorted(auto_expected_by_lane.items())}},
        "direction_counts": {"UP": int(direction_counts["UP"]), "DOWN": int(direction_counts["DOWN"]), "UNKNOWN": max(0, authoritative_total - sum(direction_counts.values()))},
        "traffic_density": {"average_active": round(avg_active, 2), "peak_active": peak_active, "level": traffic_level(round(avg_active))},
        "traffic": {"average_active_vehicles": round(avg_active, 2), "peak_active_vehicles": peak_active, "density": traffic_level(round(avg_active))},
        "vehicles_detail": final_vehicles_detail,
        "speed": {
            "average_kmh": avg_speed_kmh, "average_pixels_per_second": avg_speed_px_s,
            "speed_limit_kmh": round(speed_limit_kmh, 2), "meters_per_pixel": meters_per_pixel,
            "calibrated": bool(reference_distance_m > 0), "real_kmh_enabled": bool(reference_distance_m > 0),
            "mode": speed_mode,
            "estimate_note": "Estimated speed uses the configured meters-per-pixel scene calibration. Gate mode uses a known real-world distance and is more stable; either mode should be calibrated to the camera before enforcement use.",
        },
        "wrong_way": {"count": int(wrong_way_count), "expected_direction": expected_direction, "auto_detected_direction": auto_detected_direction, "auto_lane_expected": {str(k): v for k, v in sorted(auto_expected_by_lane.items())}, "vehicles": [v["track_id"] for v in final_vehicles_detail if v.get("wrong_way")]},
        "speed_violations": {"count": int(speed_violation_count), "limit_kmh": round(speed_limit_kmh, 2), "vehicles": [v["track_id"] for v in final_vehicles_detail if v.get("speed_violation")]},
        "lane_analysis": {"lane_count": lane_count, "lanes": lane_stats, "geometry": {"mode": "perspective-trapezoid", "top_width_ratio": lane_top_width_ratio, "bottom_width_ratio": lane_bottom_width_ratio}},
        "incidents": incidents,
        "incident_summary": {"total": len(incidents), "wrong_way": int(wrong_way_count), "speed_violation": int(speed_violation_count), "stopped_vehicle": int(stopped_count), "high_congestion": sum(1 for x in incidents if x.get("type") == "HIGH_CONGESTION"), "congestion_spike": sum(1 for x in incidents if x.get("type") == "CONGESTION_SPIKE")},
        "spatial_heatmap": {"rows": heat_rows, "columns": heat_cols, "grid": heat_grid, "hotspots": hotspots},
        "analysis_quality": {"score": quality_score, "level": quality_level, "average_detection_confidence": round(avg_conf, 3), "confirmed_track_ratio": round(confirmed_ratio, 3), "stable_track_ratio": round(stable_ratio, 3), "notes": "Analysis quality is an internal consistency signal, not a measured model accuracy/precision metric."},
        "recommendations": recommendations,
        "configuration": {"tracker": "ByteTrack-style temporal association", "detect_confidence": round(conf_thresh, 3), "frame_stride": stride, "image_size": imgsz, "nms_iou": nms_iou, "confirmation_frames": confirmation_frames, "direction_window": direction_window, "direction_min_pixels": direction_min_pixels, "speed_smoothing_window": speed_smoothing_window, "speed_margin_kmh": speed_margin_kmh, "speed_mode": speed_mode, "speed_confirmation_window": speed_confirm_window, "speed_confirmation_hits": speed_confirm_hits},
        "settings": {"expected_direction": expected_direction, "auto_detected_direction": auto_detected_direction, "auto_lane_expected": {str(k): v for k, v in sorted(auto_expected_by_lane.items())}, "speed_limit_kmh": round(speed_limit_kmh, 2), "meters_per_pixel": meters_per_pixel, "lane_count": lane_count, "stopped_seconds": stopped_seconds, "confidence": round(conf_thresh, 3), "image_size": imgsz, "frame_stride": stride, "lane_top_width_ratio": lane_top_width_ratio, "lane_bottom_width_ratio": lane_bottom_width_ratio, "speed_reference_distance_m": reference_distance_m, "speed_gate_top": gate_top, "speed_gate_bottom": gate_bottom, "speed_mode": speed_mode, "speed_confirmation_window": speed_confirm_window, "speed_confirmation_hits": speed_confirm_hits},
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if progress_callback:
        progress_callback(100, "Analysis complete")
    return report
