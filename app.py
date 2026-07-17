"""
All-in-one Gradio app for the vehicle detection + semantic search pipeline.

Replaces the old two-script workflow (vehicle_detection_with_clip.py with an
OpenCV desktop window, then a separate search_app.py) with a single Gradio
app that walks the user through:

  1. Upload a video + paste their NVIDIA NeMoVision API key
  2. Click 2 points on the first frame to draw the crossing line
  3. Click "Process video" -> vehicles get detected, tracked, cropped, sent
     to NeMoVision for brand/color/plate reading, embedded with local CLIP,
     and stored in ChromaDB. Results stream into a table live.
  4. Switch to the "Semantic search" tab to search by text or by uploading a
     car/plate photo, with a 2D embedding map (matches highlighted in red).

Run with:  python app.py
Requires license_plate_detector.pt and yolov8s.pt in the working directory.
"""

import os
import cv2
import json
import base64
import warnings
import subprocess
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

import supervision as sv
from ultralytics import YOLO

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.embed import file_html
from bokeh.resources import CDN
import html as html_lib

from local_clip import LocalCLIP
from vector_store import VehicleVectorStore

if not hasattr(sv, "BoundingBoxAnnotator"):
    sv.BoundingBoxAnnotator = sv.BoxAnnotator

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PLATE_MODEL_PATH = "license_plate_detector.pt"
VEHICLE_MODEL_PATH = "yolov8s.pt"

CAPTURES_DIR = "captures"
OUTPUTS_DIR = "outputs_app"
LOGS_DIR = "logs"

PLATE_DETECT_CONF = 0.3
PLATE_VLM_CONF_THRESHOLD = 0.6
PENDING_WINDOW_FRAMES = 25
CROP_MARGIN = 6

NVIDIA_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

VEHICLE_CLASS_IDS = {2, 3, 5, 7}

FONT = cv2.FONT_HERSHEY_SIMPLEX
VEHICLE_BOX_COLOR = (0, 200, 255)
PLATE_BOX_COLOR = (90, 230, 140)

GRADIO_PORT = 7860

for d in (CAPTURES_DIR, OUTPUTS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# FASTAPI APP (serves captures/ so Bokeh hover thumbnails can load them)
# ---------------------------------------------------------------------------
fastapi_app = FastAPI()
fastapi_app.mount("/images", StaticFiles(directory=CAPTURES_DIR), name="images")

# ---------------------------------------------------------------------------
# MODELS (loaded once at startup)
# ---------------------------------------------------------------------------
print("Loading models...")
plate_model = YOLO(PLATE_MODEL_PATH)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
VEHICLE_CLASS_NAMES = vehicle_model.names

nvclip_client = LocalCLIP()
vector_store = VehicleVectorStore()
print("Models loaded.")

TABLE_COLUMNS = ["ID", "Brand", "Color", "Plate", "Direction", "Time"]

# ---------------------------------------------------------------------------
# HELPERS (ported from the original script, cv2-GUI-free)
# ---------------------------------------------------------------------------


def _encode_jpg(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def analyze_vehicle_with_nemovision(car_crop, plate_crop, api_key):
    try:
        car_b64 = _encode_jpg(car_crop)
        plate_b64 = _encode_jpg(plate_crop)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        prompt = """You are analyzing two cropped images from a traffic camera:
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

        payload = {
            "model": NVIDIA_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{car_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{plate_b64}"}}
                ]
            }],
            "max_tokens": 250,
            "temperature": 0.2,
            "top_p": 0.95
        }

        response = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return {"vehicle_type": "unknown", "color": "unknown",
                    "license_plate_text": "unreadable", "description": f"VLM API error ({response.status_code})"}

        raw = response.json()["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start:end + 1]) if start != -1 and end != -1 else {}

        if data.get("plate_type", "").lower() == "tunisian":
            left = "".join(ch for ch in str(data.get("plate_left", "")) if ch.isdigit())
            right = "".join(ch for ch in str(data.get("plate_right", "")) if ch.isdigit())
            if left and right:
                plate_text = f"{left} TN {right}"
            elif right:
                plate_text = f"TN {right}"
            else:
                plate_text = left or "unreadable"
        else:
            plate_text = str(data.get("plate_full", "")).strip() or "unreadable"

        return {
            "vehicle_type": data.get("vehicle_type", "unknown"),
            "color": data.get("color", "unknown"),
            "license_plate_text": plate_text,
            "description": data.get("description", "")
        }

    except Exception as e:
        return {"vehicle_type": "unknown", "color": "unknown",
                "license_plate_text": "unreadable", "description": f"Error: {e}"}


def embed_and_store_vehicle(meta, info):
    try:
        car_resp = nvclip_client([meta["car_path"]])
        plate_resp = nvclip_client([meta["plate_path"]])
        car_vector = car_resp["data"][0]["embedding"]
        plate_vector = plate_resp["data"][0]["embedding"]

        vector_store.insert_vehicle(
            car_vector=car_vector,
            plate_vector=plate_vector,
            metadata={
                "track_id": meta["track_id"],
                "brand": info["vehicle_type"] or "unknown",
                "color": info["color"] or "unknown",
                "plate": info["license_plate_text"] or "unreadable",
                "direction": meta["direction"],
                "time": meta["time_str"],
                "car_image": meta["car_path"],
                "plate_image": meta["plate_path"],
                "description": info["description"],
            },
        )
    except Exception as e:
        print(f"[CLIP] embedding/insert failed for vehicle {meta['track_id']}: {e}")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / float((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def find_best_match(ref_box, candidates):
    if not candidates:
        return None
    rx1, ry1, rx2, ry2 = ref_box
    rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
    best_idx, best_score = None, 0.0
    for idx, cb in enumerate(candidates):
        cx1, cy1, cx2, cy2 = cb
        score = _iou(ref_box, cb)
        if cx1 <= rcx <= cx2 and cy1 <= rcy <= cy2:
            score += 0.5
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score > 0 else None


def safe_crop(img, box, margin=0):
    h, w = img.shape[:2]
    x1 = max(0, int(box[0]) - margin)
    y1 = max(0, int(box[1]) - margin)
    x2 = min(w, int(box[2]) + margin)
    y2 = min(h, int(box[3]) + margin)
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def line_side(px, py, p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if val > 0 else (-1 if val < 0 else 0)


def _draw_corner_box(img, x1, y1, x2, y2, color, length=18, thickness=2):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _draw_label(img, x1, y1, text, color):
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.55, 2)
    y_top = max(0, y1 - th - baseline - 8)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 10, y_top + th + baseline + 8), (15, 15, 15), -1)
    cv2.putText(img, text, (x1 + 5, y_top + th + 2), FONT, 0.55, color, 2, cv2.LINE_AA)


def draw_detections(frame, vehicle_detections, plate_detections, vehicle_display_ids, line_p1, line_p2):
    # line_p1/line_p2 are kept as arguments (still used elsewhere for the
    # crossing math) but are intentionally NOT drawn on the output video.
    annotated = frame.copy()

    if len(vehicle_detections) > 0:
        for i in range(len(vehicle_detections)):
            x1, y1, x2, y2 = vehicle_detections.xyxy[i].astype(int)
            cid = vehicle_detections.class_id[i]
            name = VEHICLE_CLASS_NAMES.get(int(cid), "vehicle").capitalize()
            disp_id = vehicle_display_ids[i]
            _draw_corner_box(annotated, x1, y1, x2, y2, VEHICLE_BOX_COLOR)
            _draw_label(annotated, x1, y1, f"{name} - ID {disp_id}", VEHICLE_BOX_COLOR)

    if len(plate_detections) > 0:
        for i in range(len(plate_detections)):
            x1, y1, x2, y2 = plate_detections.xyxy[i].astype(int)
            tid = int(plate_detections.tracker_id[i]) if plate_detections.tracker_id is not None else i
            _draw_corner_box(annotated, x1, y1, x2, y2, PLATE_BOX_COLOR)
            _draw_label(annotated, x1, y1, f"Plate - ID {tid}", PLATE_BOX_COLOR)

    return annotated


def _rows_from_log(log_list):
    return [[e["id"], e["brand"], e["color"], e["plate"], e["direction"], e["time"]] for e in log_list]


def _to_browser_h264(input_path):
    """cv2.VideoWriter's mp4v output isn't playable in most browsers, so the
    Gradio <video> player shows a blank/broken player. Re-encode to H.264
    with ffmpeg (via imageio-ffmpeg's bundled static binary -- no system
    ffmpeg install needed) so it actually plays inline. Falls back to the
    original file if ffmpeg isn't available for any reason."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        output_path = input_path.replace(".mp4", "_h264.mp4")
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        print(f"[ffmpeg] re-encode failed, serving original file: {result.stderr.decode(errors='ignore')[-500:]}")
        return input_path
    except Exception as e:
        print(f"[ffmpeg] re-encode skipped ({e}), serving original file")
        return input_path


# ---------------------------------------------------------------------------
# STEP 1: load first frame + click-to-draw crossing line
# ---------------------------------------------------------------------------


def load_first_frame(video_path):
    if not video_path:
        raise gr.Error("Upload a video first.")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise gr.Error("Could not read a frame from this video.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, rgb, [], "Click 2 points on the image to set the crossing line."


def on_image_click(orig_frame, points, evt: gr.SelectData):
    if orig_frame is None:
        raise gr.Error("Load the first frame first.")
    x, y = int(evt.index[0]), int(evt.index[1])
    points = list(points) if points else []
    if len(points) >= 2:
        points = []
    points.append((x, y))

    disp = np.array(orig_frame).copy()
    for (px, py) in points:
        cv2.circle(disp, (px, py), 7, (255, 0, 0), -1)
    if len(points) == 2:
        cv2.line(disp, points[0], points[1], (255, 0, 0), 2)
        status = f"Crossing line set: {points[0]} -> {points[1]}. Ready to process."
    else:
        status = f"First point set at {points[0]}. Click a second point."
    return disp, points, status


def reset_line(orig_frame):
    if orig_frame is None:
        raise gr.Error("Load the first frame first.")
    return np.array(orig_frame), [], "Line reset. Click 2 points on the image."


# ---------------------------------------------------------------------------
# STEP 2: process the video (generator -> streams table + status live)
# ---------------------------------------------------------------------------


def process_video(video_path, api_key, points, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    if not points or len(points) != 2:
        raise gr.Error("Draw the crossing line first (click 2 points on the frame).")
    if not api_key or not api_key.strip():
        raise gr.Error("Paste your NVIDIA NeMoVision API key first.")

    api_key = api_key.strip()
    line_p1 = tuple(int(v) for v in points[0])
    line_p2 = tuple(int(v) for v in points[1])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"vehicle_log_{run_id}.txt")
    output_video_path = os.path.join(OUTPUTS_DIR, f"processed_{run_id}.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Some libraries reset the warnings filters at import time, so re-apply here too.
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Fresh tracker per run so IDs don't carry over between videos.
    vehicle_tracker = sv.ByteTrack()
    run_plate_model = YOLO(PLATE_MODEL_PATH)  # fresh instance -> fresh internal tracker state

    frame_count = 0
    prev_side, pending, already_sent = {}, {}, set()
    in_count, out_count = 0, 0
    vehicle_log_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        progress(min(frame_count / total_frames, 1.0),
                 desc=f"Frame {frame_count}/{total_frames} | IN: {in_count}  OUT: {out_count}")

        ts = frame_count / fps
        time_str = f"{int(ts // 3600):02d}:{int((ts % 3600) // 60):02d}:{int(ts % 60):02d}"

        vresults = vehicle_model(frame, conf=0.4, verbose=False)[0]
        vehicle_detections = sv.Detections.from_ultralytics(vresults)
        if len(vehicle_detections) > 0 and vehicle_detections.class_id is not None:
            vehicle_detections = vehicle_detections[np.isin(vehicle_detections.class_id, list(VEHICLE_CLASS_IDS))]
        vehicle_detections = vehicle_tracker.update_with_detections(vehicle_detections)
        vehicle_boxes = [tuple(b) for b in vehicle_detections.xyxy] if len(vehicle_detections) > 0 else []

        presults = run_plate_model.track(frame, conf=PLATE_DETECT_CONF, persist=True, verbose=False)
        plate_detections = sv.Detections.from_ultralytics(presults[0])
        plate_boxes = [tuple(b) for b in plate_detections.xyxy] if len(plate_detections) > 0 else []

        vehicle_display_ids = []
        for i in range(len(vehicle_boxes)):
            disp_id = None
            if plate_boxes:
                pidx = find_best_match(vehicle_boxes[i], plate_boxes)
                if pidx is not None and plate_detections.tracker_id is not None:
                    disp_id = int(plate_detections.tracker_id[pidx])
            if disp_id is None:
                disp_id = int(vehicle_detections.tracker_id[i]) if vehicle_detections.tracker_id is not None else i
            vehicle_display_ids.append(disp_id)

        if len(vehicle_detections) > 0 and vehicle_detections.tracker_id is not None:
            for i, box in enumerate(vehicle_boxes):
                tid = int(vehicle_detections.tracker_id[i])
                if tid in already_sent:
                    continue
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                side = line_side(cx, cy, line_p1, line_p2)
                if tid in prev_side and prev_side[tid] != 0 and side != 0 and side != prev_side[tid]:
                    direction = "Exit" if side == 1 else "Enter"
                    pending[tid] = {"remaining": PENDING_WINDOW_FRAMES, "direction": direction}
                    if direction == "Enter":
                        in_count += 1
                    else:
                        out_count += 1
                if side != 0:
                    prev_side[tid] = side

        for tid in list(pending.keys()):
            if tid in already_sent:
                pending.pop(tid, None)
                continue

            vidx = next((i for i in range(len(vehicle_boxes))
                         if vehicle_detections.tracker_id is not None
                         and int(vehicle_detections.tracker_id[i]) == tid), None)

            matched = False
            if vidx is not None and plate_boxes:
                pidx = find_best_match(vehicle_boxes[vidx], plate_boxes)
                if pidx is not None:
                    conf = float(plate_detections.confidence[pidx]) if plate_detections.confidence is not None else 0.0
                    if conf >= PLATE_VLM_CONF_THRESHOLD:
                        plate_crop = safe_crop(frame, plate_boxes[pidx], CROP_MARGIN)
                        car_crop = safe_crop(frame, vehicle_boxes[vidx], CROP_MARGIN)
                        if plate_crop is not None and car_crop is not None:
                            plate_tid = int(plate_detections.tracker_id[pidx]) if plate_detections.tracker_id is not None else tid
                            folder = os.path.join(CAPTURES_DIR, f"{run_id}_id_{plate_tid}")
                            os.makedirs(folder, exist_ok=True)
                            car_path = os.path.join(folder, f"car_{plate_tid}.jpg")
                            plate_path = os.path.join(folder, f"plate_{plate_tid}.jpg")
                            cv2.imwrite(car_path, car_crop)
                            cv2.imwrite(plate_path, plate_crop)

                            info = analyze_vehicle_with_nemovision(car_crop, plate_crop, api_key)

                            line = (f"{time_str}  {pending[tid]['direction']}  Vehicle ID {plate_tid}  "
                                    f"{info['color']} {info['vehicle_type']}  Plate: {info['license_plate_text']}"
                                    f"  | IN: {in_count} OUT: {out_count} TOTAL: {in_count + out_count}")
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(line + "\n")

                            meta = {"frame": frame_count, "track_id": plate_tid, "conf": conf,
                                     "time_str": time_str, "direction": pending[tid]["direction"],
                                     "vehicle_folder": folder, "car_path": car_path, "plate_path": plate_path,
                                     "in_count": in_count, "out_count": out_count}
                            with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
                                json.dump({**meta, **info}, f, ensure_ascii=False, indent=2)

                            embed_and_store_vehicle(meta, info)

                            vehicle_log_list.insert(0, {
                                "id": plate_tid,
                                "brand": info["vehicle_type"] or "unknown",
                                "color": info["color"] or "unknown",
                                "plate": info["license_plate_text"] or "unreadable",
                                "time": time_str,
                                "direction": pending[tid]["direction"],
                            })

                            already_sent.add(tid)
                            matched = True

            if matched:
                pending.pop(tid, None)
            else:
                pending[tid]["remaining"] -= 1
                if pending[tid]["remaining"] <= 0:
                    pending.pop(tid, None)

        annotated = draw_detections(frame, vehicle_detections, plate_detections, vehicle_display_ids, line_p1, line_p2)
        writer.write(annotated)

    cap.release()
    writer.release()

    progress(1.0, desc="Encoding preview video...")
    playable_video_path = _to_browser_h264(output_video_path)

    status = f"Done. {len(already_sent)} vehicles logged. IN: {in_count}  OUT: {out_count}."
    return _rows_from_log(vehicle_log_list), status, playable_video_path


# ---------------------------------------------------------------------------
# STEP 3: semantic search (ported from search_app.py)
# ---------------------------------------------------------------------------

_car_ids_g = []
_coords_2d_g = None
_image_urls_g = []
_metadatas_g = []

SEARCH_TABLE_HEADERS = ["ID", "Brand", "Color", "Plate", "Direction", "Time", "Similarity"]


def _refresh_projection():
    global _coords_2d_g, _image_urls_g, _metadatas_g, _car_ids_g

    data = vector_store.all_car_vectors()
    embeddings = data.get("embeddings")
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])

    _car_ids_g = ids
    _metadatas_g = metadatas

    if embeddings is None or len(embeddings) < 2:
        _coords_2d_g = None
        _image_urls_g = []
        return

    embeddings = np.array(embeddings)
    _coords_2d_g = PCA(n_components=2).fit_transform(embeddings)
    _image_urls_g = [
        f"http://localhost:{GRADIO_PORT}/images/{Path(m['car_image']).relative_to(CAPTURES_DIR).as_posix()}"
        for m in metadatas
    ]


def highlighted_plot(highlight_ids=None):
    highlight_ids = set(highlight_ids or [])

    p = figure(
        title="Vehicle Embedding Map",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        tooltips="""
            <div>
                <div>
                    <img src="@image_path" height="100" alt="car" style="float: left; margin: 0px 15px 15px 0px;"/>
                </div>
                <div><span>@label</span></div>
            </div>
        """,
    )

    if _coords_2d_g is None:
        p.title.text = "Not enough vehicles logged yet -- process a video first."
        p.scatter([], [])  # empty renderer so Bokeh doesn't warn about a plot with none
        return p

    labels = [f"{m.get('color', '')} {m.get('brand', '')} | {m.get('plate', '')}" for m in _metadatas_g]
    source = ColumnDataSource(dict(
        x=_coords_2d_g[:, 0], y=_coords_2d_g[:, 1],
        image_path=_image_urls_g, label=labels,
    ))
    p.scatter("x", "y", source=source, size=10, color="#457b9d", legend_label="Vehicle")

    if highlight_ids:
        idxs = [i for i, _id in enumerate(_car_ids_g) if _id in highlight_ids]
        if idxs:
            hi_source = ColumnDataSource(dict(
                x=_coords_2d_g[idxs, 0], y=_coords_2d_g[idxs, 1],
                image_path=[_image_urls_g[i] for i in idxs],
                label=[labels[i] for i in idxs],
            ))
            p.scatter("x", "y", source=hi_source, size=16, color="red", legend_label="Match")

    return p


def render_bokeh_html(fig):
    doc_html = file_html(fig, CDN, "Vehicle Embedding Map")
    escaped = html_lib.escape(doc_html, quote=True)
    return f'<iframe srcdoc="{escaped}" style="width:100%;height:560px;border:none;"></iframe>'


def _build_table_rows(hits):
    rows = []
    for h in hits:
        e = h["entity"]
        similarity = round(1 - h["distance"], 3)
        rows.append([e["track_id"], e["brand"], e["color"], e["plate"], e["direction"], e["time"], similarity])
    return rows


def query_callback(text_query, image_path, photo_type):
    _refresh_projection()

    if image_path is not None:
        resp = nvclip_client([image_path])
        vector = resp["data"][0]["embedding"]
        if photo_type == "Plate":
            hits = vector_store.search_plates(vector)
            highlight_ids = ["car_" + h["id"].split("_", 1)[1] for h in hits]
        else:
            hits = vector_store.search_cars(vector)
            highlight_ids = [h["id"] for h in hits]
    elif text_query and text_query.strip():
        resp = nvclip_client([text_query.strip()])
        vector = resp["data"][0]["embedding"]
        hits = vector_store.search_cars(vector)
        highlight_ids = [h["id"] for h in hits]
    else:
        return [], render_bokeh_html(highlighted_plot()), [], [], "Search to see vehicle details here."

    gallery_items = [
        (h["entity"]["car_image"],
         f"#{i + 1} - {h['entity']['color']} {h['entity']['brand']} | sim: {round(1 - h['distance'], 3)}")
        for i, h in enumerate(hits)
    ]
    return (gallery_items, render_bokeh_html(highlighted_plot(highlight_ids)), _build_table_rows(hits),
            hits, "Click a result above to see its details here.")


def show_selected_details(hits, evt: gr.SelectData):
    if not hits or evt.index is None or evt.index >= len(hits):
        return "Click a result above to see its details here."
    e = hits[evt.index]["entity"]
    similarity = round(1 - hits[evt.index]["distance"], 3)
    direction_word = "Entered" if e.get("direction") == "Enter" else "Exited"
    return (
        f"### Result #{evt.index + 1}\n"
        f"- **Vehicle ID:** {e.get('track_id', '?')}\n"
        f"- **Brand:** {e.get('brand', 'unknown')}\n"
        f"- **Color:** {e.get('color', 'unknown')}\n"
        f"- **Plate:** {e.get('plate', 'unreadable')}\n"
        f"- **{direction_word} the frame at:** {e.get('time', '?')}\n"
        f"- **Similarity:** {similarity}\n"
        f"- **Description:** {e.get('description', '')}"
    )


def refresh_map_only():
    _refresh_projection()
    return render_bokeh_html(highlighted_plot())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Monochrome(), title="Tunisian Vehicle Search Using VLMs and CLIP") as blocks:
    gr.HTML(
        '<div style="text-align:center;">'
        '<h1 style="color:#e8720c;font-weight:800;font-size:230%;margin-bottom:0;">Tunisian Vehicle Search Using VLMs and CLIP</h1>'
        '<p style="color:#888;margin-top:4px;">Developed by Wassim Hfaiedh</p>'
        "</div>"
    )

    state_orig_frame = gr.State(None)
    state_points = gr.State([])

    with gr.Tab("1. Setup & crossing line"):
        with gr.Row():
            video_input = gr.File(label="Upload traffic video", type="filepath",
                                   file_types=[".mp4", ".avi", ".mov", ".mkv"])
            api_key_input = gr.Textbox(label="NVIDIA NeMoVision API key", type="password",
                                        placeholder="nvapi-...")
        load_frame_btn = gr.Button("Load first frame")
        frame_image = gr.Image(label="Click 2 points on the image to draw the crossing line", type="numpy")
        with gr.Row():
            reset_line_btn = gr.Button("Reset line")
        line_status = gr.Textbox(label="Status", interactive=False)

        load_frame_btn.click(load_first_frame, inputs=[video_input],
                              outputs=[frame_image, state_orig_frame, state_points, line_status])
        frame_image.select(on_image_click, inputs=[state_orig_frame, state_points],
                            outputs=[frame_image, state_points, line_status])
        reset_line_btn.click(reset_line, inputs=[state_orig_frame],
                              outputs=[frame_image, state_points, line_status])

    with gr.Tab("2. Process video"):
        process_btn = gr.Button("Process video", variant="primary")
        progress_status = gr.Textbox(label="Status", interactive=False)
        output_video = gr.Video(label="Annotated output video")
        results_table = gr.Dataframe(headers=TABLE_COLUMNS, label="Detected vehicles", wrap=True)

        process_btn.click(process_video, inputs=[video_input, api_key_input, state_points],
                           outputs=[results_table, progress_status, output_video], show_progress="minimal")

    with gr.Tab("3. Semantic search"):
        gr.HTML("<h3>Search by description, or upload a car / plate photo.</h3>")
        with gr.Row():
            text_query = gr.Textbox(placeholder="e.g. silver peugeot, black car, red truck", show_label=False)
            image_upload = gr.Image(type="filepath", label="Or upload a photo")
            photo_type = gr.Radio(["Car", "Plate"], value="Car", label="Uploaded photo is a...")
        refresh_btn = gr.Button("Refresh embedding map")

        state_hits = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                gallery = gr.Gallery(columns=4, height=650, object_fit="scale-down", preview=False,
                                      label="Results (click one to see its details)")
                selected_details = gr.Markdown("Click a result above to see its details here.")
            embedding_plot = gr.HTML(value=render_bokeh_html(highlighted_plot()))

        details_table = gr.Dataframe(headers=SEARCH_TABLE_HEADERS, label="Vehicle details")

        text_query.change(query_callback, [text_query, image_upload, photo_type],
                           [gallery, embedding_plot, details_table, state_hits, selected_details], show_progress=False)
        image_upload.upload(query_callback, [text_query, image_upload, photo_type],
                             [gallery, embedding_plot, details_table, state_hits, selected_details], show_progress=False)
        refresh_btn.click(refresh_map_only, outputs=[embedding_plot])
        gallery.select(show_selected_details, inputs=[state_hits], outputs=[selected_details])


def main():
    app = gr.mount_gradio_app(fastapi_app, blocks, path="/")

    print(f"Access the app at http://localhost:{GRADIO_PORT}")
    uvicorn.run(app, host="localhost", port=GRADIO_PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()