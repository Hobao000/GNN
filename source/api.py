from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source"

# Cho phép các file trong source/ import kiểu:
# from data_loader import ...
# from losses import ...
# from models.gnn_models import ...
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from predict import predict_csv

MODEL_PATH = ROOT / "data/models/graphsage_focal/gnn_graphsage_bundle.pth"
UPLOAD_DIR = ROOT / "data/uploads"
OUTPUT_DIR = ROOT / "data/web_outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AML GNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo/hi-small")
def hi_small_demo():
    summary_path = ROOT / "data/outputs_hi_small/inference_summary.json"
    graph_path = ROOT / "data/outputs_hi_small/web_graph.json"
    metrics_path = ROOT / "data/outputs_hi_small/evaluation_metrics.json"

    if not summary_path.exists() or not graph_path.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "Chưa có output HI-Small. Hãy chạy predict trước."},
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return {
        "summary": summary,
        "metrics": metrics,
        "graph": graph,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    upload_path = UPLOAD_DIR / file.filename

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_dir = OUTPUT_DIR / upload_path.stem

    result = predict_csv(
        bundle_path=MODEL_PATH,
        csv_path=upload_path,
        output_dir=output_dir,
    )

    graph_path = output_dir / "web_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    return {
        "summary": result,
        "metrics": None,
        "graph": graph,
    }