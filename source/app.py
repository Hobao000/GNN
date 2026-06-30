from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_csv

app = FastAPI(title="AML GNN Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_BUNDLE = Path("data/models/gnn_graphsage_bundle.pth")
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")


@app.get("/")
def root():
    return {"message": "AML GNN API is running"}


@app.post("/predict")
def predict(file: UploadFile = File(...), bundle_path: str = str(DEFAULT_BUNDLE)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    csv_path = UPLOAD_DIR / f"uploaded_transactions{suffix}"
    with csv_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = predict_csv(bundle_path=bundle_path, csv_path=csv_path, output_dir=OUTPUT_DIR)
    return result
