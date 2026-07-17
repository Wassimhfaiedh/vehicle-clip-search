# Tunisian Vehicle Search Using VLMs and CLIP



This project uses object detection (YOLO) and a vision-language model (VLM)
to detect vehicles and read their license plates exactly as printed —
including Tunisian-style plates (digits + "TN" + digits). Every detected
vehicle is also embedded with CLIP, so you can search your logged vehicles
using natural language ("silver peugeot") or by uploading a photo of a car
or a license plate.

## Setup

1. Put these two files in the same folder as `app.py`:
   - `license_plate_detector.pt`
   - `yolov8s.pt`
2. `pip install -r requirements.txt`
3. `python app.py`
4. Open `http://localhost:7860`

## Flow (3 tabs)

1. **Setup & crossing line** — upload the video, paste your NVIDIA NeMoVision
   API key, click "Load first frame", then click 2 points on the image to
   draw the crossing line (click a 3rd time to restart the line).
2. **Process video** — click "Process video". A progress bar tracks the
   frames as vehicles crossing the line get cropped, sent to NeMoVision for
   brand/color/plate reading, and embedded with local CLIP into ChromaDB.
   When done, the annotated output video and the results table appear.
3. **Semantic search** — search by text ("silver peugeot") or upload a car
   or plate photo. Matches show as a numbered gallery — click any result to
   see its full details (brand, color, plate, enter/exit time). A details
   table and a 2D embedding map (matches highlighted in red) are also shown.

## Notes

- The API key field is left empty on purpose — **the key that was hardcoded
  in the original script was exposed in shared code and should be revoked
  in the NVIDIA console**, then a fresh key pasted into the app each session
  (or set as an env var yourself if you'd rather not paste it each time).
- Captures are saved under `captures/<run_id>_id_<vehicle_id>/`, so runs
  from different videos don't collide.
- ChromaDB data persists in `vehicle_search_chroma/` between runs, so
  vehicles logged in past sessions stay searchable.