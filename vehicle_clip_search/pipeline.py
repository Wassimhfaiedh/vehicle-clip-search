"""Headless vehicle detection pipeline: YOLO detection + ByteTrack tracking +
line-crossing + Nemotron VLM analysis + CLIP embedding + ChromaDB storage.
Called synchronously from the Gradio app, once per "Process" click, and
returns the list of logged vehicles."""

import base64
import json
import os

import cv2
import numpy as np
import requests
import supervision as sv
from ultralytics import YOLO

from . import clip_embedder, vector_store
from .config import (
    PLATE_MODEL_PATH, VEHICLE_MODEL_PATH, VEHICLE_CLASS_IDS, CROP_MARGIN,
    PLATE_DETECT_CONF, PLATE_VLM_CONF_THRESHOLD, PENDING_WINDOW_FRAMES,
    CAPTURES_DIR, NVIDIA_MODEL, NVIDIA_URL,
)

if not hasattr(sv, "BoundingBoxAnnotator"):
    sv.BoundingBoxAnnotator = sv.BoxAnnotator

_vehicle_model = None
_plate_model = None


def _load_models():
    global _vehicle_model, _plate_model
    if _vehicle_model is None:
        _vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    if _plate_model is None:
        _plate_model = YOLO(PLATE_MODEL_PATH)


VLM_PROMPT = """You are analyzing two cropped images from a traffic camera:
image 1 is the full vehicle, image 2 is a close-up of its license plate.

First decide the plate type:
- "tunisian": plate uses Arabic script / Tunisia's "TN" style, printed as two separate number
  groups with a divider in the middle (e.g. "128" TN "78").
- "foreign": plate uses Latin letters (English, French, or any non-Arabic script), like a normal
  European/international plate (e.g. "AB-123-CD").

Respond ONLY with valid JSON, no markdown, no backticks, no extra text:

{
  "vehicle_type": "<car brand/make if visible, e.g. BMW, Peugeot, Toyota, Renault, unknown>",
  "color": "<dominant color of the vehicle>",
  "plate_type": "<tunisian|foreign>",
  "plate_left": "<if tunisian: digits left of the divider, exactly as shown, else ''>",
  "plate_right": "<if tunisian: digits right of the divider, exactly as shown, else ''>",
  "plate_full": "<if foreign: the full plate exactly as printed (letters+digits, keep its own spacing/dashes), else ''>",
  "description": "<one full sentence describing the vehicle: brand, color, and plate>"
}

Read each side/field independently and exactly as printed - do not force a fixed digit count."""


def _encode_jpg(img) -> str:
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def analyze_vehicle(car_crop, plate_crop, api_key: str) -> dict:
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_jpg(car_crop)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_jpg(plate_crop)}"}},
            ],
        }],
        "max_tokens": 250,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])

    if data.get("plate_type", "").lower() == "tunisian":
        left = "".join(c for c in str(data.get("plate_left", "")) if c.isdigit())
        right = "".join(c for c in str(data.get("plate_right", "")) if c.isdigit())
        plate_text = f"{left} TN {right}" if left and right else (f"TN {right}" if right else (left or "unreadable"))
    else:
        plate_text = str(data.get("plate_full", "")).strip() or "unreadable"

    return {
        "vehicle_type": data.get("vehicle_type", "unknown"),
        "color": data.get("color", "unknown"),
        "license_plate_text": plate_text,
        "description": data.get("description", ""),
    }


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / float((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def _find_best_match(ref_box, candidates, exclude=frozenset()):
    if not candidates:
        return None
    rx1, ry1, rx2, ry2 = ref_box
    rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
    best_idx, best_score = None, 0.0
    for idx, cb in enumerate(candidates):
        if idx in exclude:
            continue
        cx1, cy1, cx2, cy2 = cb
        score = _iou(ref_box, cb)
        if cx1 <= rcx <= cx2 and cy1 <= rcy <= cy2:
            score += 0.5
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score > 0 else None


def _safe_crop(img, box, margin=0):
    h, w = img.shape[:2]
    x1 = max(0, int(box[0]) - margin)
    y1 = max(0, int(box[1]) - margin)
    x2 = min(w, int(box[2]) + margin)
    y2 = min(h, int(box[3]) + margin)
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def _line_side(px, py, p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if val > 0 else (-1 if val < 0 else 0)


VEHICLE_BOX_COLOR = (0, 200, 255)
PLATE_BOX_COLOR = (90, 230, 140)
LINE_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_box(frame, box, color, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_frame(frame, line_p1, line_p2, vehicle_boxes, vehicle_ids, plate_boxes, plate_ids):
    annotated = frame.copy()
    cv2.line(annotated, tuple(map(int, line_p1)), tuple(map(int, line_p2)), LINE_COLOR, 2)
    for box, vid in zip(vehicle_boxes, vehicle_ids):
        _draw_box(annotated, box, VEHICLE_BOX_COLOR, f"Vehicle {vid}")
    for box, pid in zip(plate_boxes, plate_ids):
        _draw_box(annotated, box, PLATE_BOX_COLOR, f"Plate {pid}")
    return annotated


def get_first_frame(video_path: str):
    """RGB numpy array of the first frame, for the Gradio line-picker preview."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Cannot read video")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _finalize_vehicle(car_crop, plate_crop, track_id, time_str, direction,
                       in_count, out_count, api_key) -> dict:
    info = analyze_vehicle(car_crop, plate_crop, api_key)

    folder = os.path.join(CAPTURES_DIR, f"id_{track_id}")
    os.makedirs(folder, exist_ok=True)
    car_path = os.path.join(folder, f"car_{track_id}.jpg")
    plate_path = os.path.join(folder, f"plate_{track_id}.jpg")
    cv2.imwrite(car_path, car_crop)
    cv2.imwrite(plate_path, plate_crop)

    metadata = {
        "track_id": track_id, "time": time_str, "direction": direction,
        "in_count": in_count, "out_count": out_count,
        "brand": info["vehicle_type"], "color": info["color"],
        "plate": info["license_plate_text"], "description": info["description"],
        "car_image": car_path, "plate_image": plate_path,
    }
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    car_vector = clip_embedder.embed_image(car_path)
    plate_vector = clip_embedder.embed_image(plate_path)
    vector_store.insert_vehicle(car_vector, plate_vector, metadata)
    return metadata


def process_video(video_path: str, line_p1, line_p2, api_key: str, progress_cb=None):
    """Runs the full pipeline on a video. Returns (annotated_video_path, records).
    Crops + embeddings are persisted as a side effect, so they become
    searchable right after this returns."""
    _load_models()
    os.makedirs(CAPTURES_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = sv.ByteTrack()

    output_path = os.path.join(CAPTURES_DIR, "annotated_output.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    frame_count = 0
    prev_side, pending, already_sent = {}, {}, set()
    finalized_plate_ids = set()
    in_count = out_count = 0
    records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if progress_cb:
            progress_cb(min(frame_count / total_frames, 1.0))

        ts = frame_count / fps
        time_str = f"{int(ts // 3600):02d}:{int((ts % 3600) // 60):02d}:{int(ts % 60):02d}"

        vresults = _vehicle_model(frame, conf=0.4, verbose=False)[0]
        vehicle_detections = sv.Detections.from_ultralytics(vresults)
        if len(vehicle_detections) > 0 and vehicle_detections.class_id is not None:
            vehicle_detections = vehicle_detections[np.isin(vehicle_detections.class_id, list(VEHICLE_CLASS_IDS))]
        vehicle_detections = tracker.update_with_detections(vehicle_detections)
        vehicle_boxes = [tuple(b) for b in vehicle_detections.xyxy] if len(vehicle_detections) > 0 else []

        presults = _plate_model.track(frame, conf=PLATE_DETECT_CONF, persist=True, verbose=False)
        plate_detections = sv.Detections.from_ultralytics(presults[0])
        plate_boxes = [tuple(b) for b in plate_detections.xyxy] if len(plate_detections) > 0 else []

        vehicle_ids = ([int(t) for t in vehicle_detections.tracker_id]
                        if len(vehicle_detections) > 0 and vehicle_detections.tracker_id is not None
                        else list(range(len(vehicle_boxes))))
        plate_ids = ([int(t) for t in plate_detections.tracker_id]
                     if len(plate_detections) > 0 and plate_detections.tracker_id is not None
                     else list(range(len(plate_boxes))))
        writer.write(_draw_frame(frame, line_p1, line_p2, vehicle_boxes, vehicle_ids, plate_boxes, plate_ids))

        if len(vehicle_detections) > 0 and vehicle_detections.tracker_id is not None:
            for i, box in enumerate(vehicle_boxes):
                tid = int(vehicle_detections.tracker_id[i])
                if tid in already_sent:
                    continue
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                side = _line_side(cx, cy, line_p1, line_p2)
                if tid in prev_side and prev_side[tid] != 0 and side != 0 and side != prev_side[tid]:
                    direction = "Exit" if side == 1 else "Enter"
                    pending[tid] = {"remaining": PENDING_WINDOW_FRAMES, "direction": direction}
                    in_count += direction == "Enter"
                    out_count += direction == "Exit"
                if side != 0:
                    prev_side[tid] = side

        used_plate_indices = set()
        for tid in list(pending.keys()):
            if tid in already_sent:
                pending.pop(tid, None)
                continue

            vidx = next((i for i in range(len(vehicle_boxes))
                         if vehicle_detections.tracker_id is not None
                         and int(vehicle_detections.tracker_id[i]) == tid), None)

            matched = False
            if vidx is not None and plate_boxes:
                pidx = _find_best_match(vehicle_boxes[vidx], plate_boxes, exclude=used_plate_indices)
                if pidx is not None:
                    conf = float(plate_detections.confidence[pidx]) if plate_detections.confidence is not None else 0.0
                    if conf >= PLATE_VLM_CONF_THRESHOLD:
                        plate_tid = (int(plate_detections.tracker_id[pidx])
                                     if plate_detections.tracker_id is not None else tid)
                        if plate_tid in finalized_plate_ids:
                            used_plate_indices.add(pidx)
                            already_sent.add(tid)
                            matched = True
                        else:
                            plate_crop = _safe_crop(frame, plate_boxes[pidx], CROP_MARGIN)
                            car_crop = _safe_crop(frame, vehicle_boxes[vidx], CROP_MARGIN)
                            if plate_crop is not None and car_crop is not None:
                                record = _finalize_vehicle(
                                    car_crop, plate_crop, plate_tid, time_str,
                                    pending[tid]["direction"], in_count, out_count, api_key,
                                )
                                records.append(record)
                                already_sent.add(tid)
                                finalized_plate_ids.add(plate_tid)
                                used_plate_indices.add(pidx)
                                matched = True

            if matched:
                pending.pop(tid, None)
            else:
                pending[tid]["remaining"] -= 1
                if pending[tid]["remaining"] <= 0:
                    pending.pop(tid, None)

    cap.release()
    writer.release()
    return output_path, records
