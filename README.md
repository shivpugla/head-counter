# HeadCount — AI Classroom Student Counter

A full-stack web application that uses **YOLOv8** to count students in classroom images.

## Architecture

```
headcount/
├── backend/          # FastAPI + YOLOv8 detection engine
│   ├── main.py
│   └── requirements.txt
├── frontend/         # React (Vite) premium UI
│   ├── src/
│   └── ...
└── start.ps1         # One-click launcher (PowerShell)
```

## Quick Start

### 1. Backend

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

> On first run, YOLOv8 will automatically download the model weights (~130 MB for yolov8x).

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5173** in your browser.

## Features

| Feature | Details |
|---|---|
| **Detection Engine** | YOLOv8x (person class) with adjustable confidence & NMS/IoU thresholds |
| **Image Formats** | JPG, PNG |
| **Annotated Output** | Bounding boxes with confidence scores + student count badge |
| **Export** | Download annotated image (PNG), JSON log, or CSV report |
| **Threshold Sliders** | Real-time confidence (default 0.35) and IoU (default 0.45) controls |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/detect` | Upload image + run detection |
| `GET` | `/download/image/{id}` | Download annotated image |
| `GET` | `/download/json/{id}` | Download JSON log |
| `GET` | `/download/csv/{id}` | Download CSV report |
| `GET` | `/health` | Health check |
