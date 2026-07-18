# Tunisian Vehicle Search

> **VLM + CLIP Vehicle Surveillance Platform** — Detect, read plates, and semantically search vehicles from traffic video.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio) ![YOLO](https://img.shields.io/badge/YOLOv8-Detection-purple) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange) ![CLIP](https://img.shields.io/badge/CLIP-Embeddings-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

This project uses object detection (YOLO) and a vision-language model (VLM) to detect
vehicles and read their license plates exactly as printed — including Tunisian-style
plates (digits + "TN" + digits). Every detected vehicle is also embedded with CLIP, so
you can search your logged vehicles using natural language ("silver peugeot") or by
uploading a photo of a car or a license plate.

---

## 🎬 Demo

**General Overview**

<img src="assets/TunisianVehicleSearch-ezgif.com-video-to-gif-converter.gif" width="900"/>

**YOLO Detection + VLM Plate Extraction**

<img src="assets/v1_annotated-ezgif.com-video-to-gif-converter.gif" width="900"/>

<table>
  <tr>
    <td align="center" colspan="2"><b>Detected Vehicles</b></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/car_8_annotated.jpg" width="400"/></td>
    <td align="center" width="50%"><img src="assets/car_4_annotated.jpg" width="400"/></td>
  </tr>
  <tr>
    <td align="center">silver Peugeot : 135 TN 7566</td>
    <td align="center">white Peugeot van : 135 TN 9434</td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><b>License Plate Reads</b></td>
  </tr>
  <tr>
    <td>
      <img src="assets/plate_5_annotated.jpg" width="400"/>
      <img src="assets/plate_21_annotated.jpg" width="400"/>
    </td>
  </tr>
</table>

---

## ✨ Features

| Feature | Description |
|---|---|
|  **Vehicle Detection** | YOLOv8 detects vehicles per frame, ByteTrack keeps a stable ID across the video |
|  **Line-Crossing Counter** | Draw a line on the first frame; vehicles are logged only when they cross it, tagged Enter/Exit |
|  **Plate Reading via VLM** | NVIDIA Nemotron reads brand, color, and plate text exactly as printed — Tunisian and foreign formats |
|  **CLIP Embeddings** | Every logged car and plate crop is embedded for semantic retrieval |
|  **Semantic Search** | Search your logged vehicles with natural language ("silver peugeot") or an uploaded photo |
|  **Vector Storage** | ChromaDB persists car/plate embeddings and metadata locally, no external DB needed |

---

## 🏗️ Architecture

```
vehicle-clip-search/
├── app.py                        # Gradio UI (process video + semantic search)
├── vehicle_clip_search/
│   ├── config.py                 # settings, loaded from .env
│   ├── pipeline.py               # YOLO + ByteTrack + VLM + CLIP + storage
│   ├── clip_embedder.py          # open_clip image/text embeddings
│   └── vector_store.py           # ChromaDB read/write
├── assets/                       # demo media (gifs, screenshots)
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An NVIDIA API key for the Nemotron VLM ([build.nvidia.com](https://build.nvidia.com))
- YOLO weights: `yolov8s.pt` and a license-plate detector `license_plate_detector.pt`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Wassimhfaiedh/TunisianVehicleSearch.git
cd TunisianVehicleSearch

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at **http://127.0.0.1:7860**

---

## ⚙️ Environment Variables

Create a `.env` file at the project root:

```env
NVIDIA_API_KEY=your-nvidia-api-key-here
PLATE_MODEL_PATH=license_plate_detector.pt
VEHICLE_MODEL_PATH=yolov8s.pt
CAPTURES_DIR=captures
DB_PATH=vehicle_search_chroma
```

---

## 📋 Requirements

```
gradio
opencv-python
numpy
requests
supervision
ultralytics
torch
open_clip_torch
Pillow
chromadb
python-dotenv
```

---

## 🖥️ Usage

### 1. Process Video tab
1. Upload a video
2. Click 2 points on the frame to set the crossing line
3. Enter your Nemotron API key (or set it via `.env`)
4. Click **Process** — each vehicle that crosses the line is detected, plate-read, embedded, and logged

### 2. Semantic Search tab
- Type a natural language query (e.g. `"silver peugeot"`), or
- Upload a photo of a car or a plate to find visual matches
- Click any result to see full vehicle details

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — vehicle & plate detection
- [supervision](https://github.com/roboflow/supervision) — ByteTrack tracking
- [NVIDIA NIM](https://build.nvidia.com/) — Nemotron VLM inference
- [open_clip](https://github.com/mlfoundations/open_clip) — CLIP embeddings
- [ChromaDB](https://www.trychroma.com/) — vector store

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
