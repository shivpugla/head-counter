"""
Classroom Headcount API
========================
FastAPI backend that accepts classroom images, runs YOLOv8 person/head detection,
and returns an annotated image with bounding boxes plus a JSON summary.
"""

import io
import uuid
import datetime
import json
import csv
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# App & model setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Classroom Headcount API",
    version="1.0.0",
    description="Upload a classroom image and get a student head count.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup – defaults to YOLOv8x (best accuracy).
# Falls back to yolov8n if the larger weights haven't been downloaded yet.
MODEL_PATH = "yolov8x.pt"
model: Optional[YOLO] = None

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# COCO class index for "person"
PERSON_CLASS_ID = 0

# Color palette for bounding boxes (BGR for OpenCV)
BOX_COLOR = (0, 220, 120)       # vivid green
TEXT_BG_COLOR = (30, 30, 30)     # dark background for labels
TEXT_COLOR = (255, 255, 255)     # white text


@app.on_event("startup")
async def load_model():
    """Load the YOLO model at startup so the first request isn't slow."""
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"[✓] Loaded model: {MODEL_PATH}")
    except Exception:
        # Fallback to the nano model if the large model isn't available
        model = YOLO("yolov8n.pt")
        print("[⚠] Fell back to yolov8n.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes → OpenCV BGR image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def _annotate(image: np.ndarray, boxes, confs, scale: float = 1.0) -> np.ndarray:
    """Draw bounding boxes + confidence labels on the image."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    # Dynamic font scale based on image size
    font_scale = max(0.4, min(w, h) / 1200) * scale
    thickness = max(1, int(font_scale * 2))

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        # Label
        label = f"{conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Text background
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), TEXT_BG_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, thickness)

    # Count badge in top-left
    badge = f"Students: {len(boxes)}"
    badge_scale = font_scale * 1.6
    badge_thick = max(2, int(badge_scale * 2))
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, badge_scale, badge_thick)
    cv2.rectangle(annotated, (10, 10), (bw + 30, bh + 30), (20, 20, 20), -1)
    cv2.rectangle(annotated, (10, 10), (bw + 30, bh + 30), BOX_COLOR, 2)
    cv2.putText(annotated, badge, (20, bh + 20),
                cv2.FONT_HERSHEY_SIMPLEX, badge_scale, BOX_COLOR, badge_thick)

    return annotated


def _image_to_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode OpenCV image to bytes."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed.")
    return buffer.tobytes()


def _save_log(record: dict) -> Path:
    """Persist detection record as JSON in the logs directory."""
    log_path = LOGS_DIR / f"{record['id']}.json"
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return log_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/detect")
async def detect_students(
    file: UploadFile = File(...),
    confidence: float = Query(0.35, ge=0.05, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.05, le=1.0, description="IoU / NMS threshold"),
):
    """
    Run YOLOv8 person detection on an uploaded classroom image.

    Returns JSON with the count, per-detection data, and a unique session ID
    that can be used to fetch the annotated image or export CSV/JSON.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are accepted.")

    contents = await file.read()
    image = _read_image(contents)

    # Inference – filter to person class only
    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    # Build annotated image
    annotated = _annotate(image, boxes, confs)
    annotated_bytes = _image_to_bytes(annotated)

    # Create log record
    session_id = uuid.uuid4().hex[:12]
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    detections = [
        {
            "bbox": [float(x) for x in box],
            "confidence": float(c),
        }
        for box, c in zip(boxes, confs)
    ]

    record = {
        "id": session_id,
        "timestamp": timestamp,
        "filename": file.filename,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "student_count": len(boxes),
        "detections": detections,
    }
    _save_log(record)

    # Store annotated image temporarily for download
    annotated_path = LOGS_DIR / f"{session_id}.png"
    with open(annotated_path, "wb") as f:
        f.write(annotated_bytes)

    import base64
    annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

    return JSONResponse(content={
        **record,
        "annotated_image_b64": annotated_b64,
    })


@app.get("/download/image/{session_id}")
async def download_annotated_image(session_id: str):
    """Download the annotated image for a given session."""
    path = LOGS_DIR / f"{session_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    return StreamingResponse(
        open(path, "rb"),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=headcount_{session_id}.png"},
    )


@app.get("/download/json/{session_id}")
async def download_json_log(session_id: str):
    """Download the JSON log for a given session."""
    path = LOGS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f"attachment; filename=headcount_{session_id}.json"},
    )


@app.get("/download/csv/{session_id}")
async def download_csv_log(session_id: str):
    """Download the detection log as CSV for a given session."""
    path = LOGS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    with open(path) as f:
        data = json.load(f)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["session_id", "timestamp", "filename", "student_count",
                      "confidence_threshold", "iou_threshold",
                      "detection_index", "x1", "y1", "x2", "y2", "confidence"])

    for i, det in enumerate(data.get("detections", [])):
        writer.writerow([
            data["id"], data["timestamp"], data["filename"], data["student_count"],
            data["confidence_threshold"], data["iou_threshold"],
            i + 1, *det["bbox"], det["confidence"],
        ])

    # If no detections, still write a summary row
    if not data.get("detections"):
        writer.writerow([
            data["id"], data["timestamp"], data["filename"], 0,
            data["confidence_threshold"], data["iou_threshold"],
            "", "", "", "", "", "",
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=headcount_{session_id}.csv"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
