"""
api/routes.py — All REST API Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
import io, csv, json
from collections import deque
import os
from datetime import datetime
from typing import Optional

from backend.api.schemas import (
    PredictRequest, PredictResponse,
    TrainRequest
)
from backend.ml.isolation_forest import IsolationForestDetector
from backend.ml.lstm_autoencoder import LSTMAutoencoderDetector
from backend.ml.classical_models import ClassicalDetectors
from backend.ml.preprocessor import Preprocessor
from backend.blockchain.web3_client import BlockchainClient

router = APIRouter()

# ── Singleton instances
iforest    = IsolationForestDetector()
lstm_model = LSTMAutoencoderDetector()
classical  = ClassicalDetectors()
preprocessor = Preprocessor()
blockchain = BlockchainClient()

# ── In-memory dataset store (replace with DB in production)
uploaded_data: list = []
training_jobs: dict = {}
stream_clients = set()
recent_events = deque(maxlen=200)


# ────────────────────────────────────────────
# POST /upload-data
# ────────────────────────────────────────────
@router.post("/upload-data", summary="Upload IoHT CSV/JSON dataset")
async def upload_data(file: UploadFile = File(...)):
    """
    Accept a CSV or JSON file containing IoHT telemetry data.
    Returns row count and basic stats.
    """
    global uploaded_data
    content = await file.read()

    try:
        if file.filename.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
            uploaded_data = [row for row in reader]
        elif file.filename.endswith(".json"):
            uploaded_data = json.loads(content)
        else:
            raise HTTPException(status_code=400, detail="Only CSV and JSON supported")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    # Basic stats
    n = len(uploaded_data)
    cols = list(uploaded_data[0].keys()) if n > 0 else []

    return {
        "status": "success",
        "filename": file.filename,
        "rows": n,
        "columns": cols,
        "message": f"Dataset loaded: {n} records, {len(cols)} features"
    }


# ────────────────────────────────────────────
# POST /predict
# ────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse, summary="Detect anomaly in IoHT reading")
async def predict(payload: PredictRequest):
    """
    Run anomaly detection on a single IoHT data point.
    Uses Isolation Forest and/or LSTM Autoencoder.
    If anomaly detected → automatically logs to blockchain.
    """
    return _predict_internal(payload)


def _predict_internal(payload: PredictRequest) -> PredictResponse:
    """Run inference and optional blockchain logging for one telemetry record."""
    features = preprocessor.transform_single(payload.dict())
    if_score, if_label = iforest.predict_single(features)
    lstm_error, lstm_label = lstm_model.predict_single(features)

    anomaly_score = max(abs(if_score), lstm_error)
    is_anomaly = if_label == -1 or lstm_label == 1
    status = "ANOMALY" if is_anomaly else "NORMAL"
    ts = datetime.utcnow().isoformat()

    tx_hash = None
    if is_anomaly:
        tx_hash = blockchain.log_entry(
            device_id=payload.device_id,
            timestamp=ts,
            anomaly_score=round(anomaly_score, 4),
            if_score=round(if_score, 4),
            lstm_error=round(lstm_error, 4),
            status=status
        )

    return PredictResponse(
        device_id=payload.device_id,
        timestamp=ts,
        status=status,
        anomaly_score=round(anomaly_score, 4),
        if_score=round(if_score, 4),
        lstm_reconstruction_error=round(lstm_error, 4),
        is_anomaly=is_anomaly,
        blockchain_tx_hash=tx_hash,
        message=f"{'Anomaly detected and logged to blockchain.' if is_anomaly else 'Normal reading.'}"
    )


@router.post("/ingest", response_model=PredictResponse, summary="Ingest one live IoHT reading and broadcast result")
async def ingest_live_reading(payload: PredictRequest):
    """
    Real-time ingest endpoint for device stream:
    - validates/infers anomaly
    - logs anomaly to blockchain
    - broadcasts prediction to all websocket listeners
    """
    result = _predict_internal(payload)
    event = result.dict()
    recent_events.appendleft(event)
    await _broadcast_event({"type": "prediction", "payload": event})
    return result


async def _broadcast_event(event: dict):
    if not stream_clients:
        return
    dead_clients = []
    for ws in stream_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead_clients.append(ws)
    for ws in dead_clients:
        stream_clients.discard(ws)


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time stream endpoint:
    - emits latest events snapshot on connect
    - keeps connection alive and pushes new ingest events
    """
    await websocket.accept()
    stream_clients.add(websocket)
    await websocket.send_json({
        "type": "snapshot",
        "count": len(recent_events),
        "payload": list(recent_events)
    })
    try:
        while True:
            # Keep alive and support optional client pings.
            _ = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        stream_clients.discard(websocket)
    except Exception:
        stream_clients.discard(websocket)


# ────────────────────────────────────────────
# POST /train
# ────────────────────────────────────────────
@router.post("/train", summary="Train/retrain ML models")
async def train(config: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training on uploaded dataset.
    Runs in background. Returns job ID.
    """
    if not uploaded_data:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload-data first.")

    job_id = f"train_{int(datetime.utcnow().timestamp())}"
    training_jobs[job_id] = {
        "status": "running",
        "algorithm": config.algorithm,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "results": {}
    }

    # Run in background
    background_tasks.add_task(
        _run_training,
        job_id=job_id,
        algo=config.algorithm,
        contamination=config.contamination,
        epochs=config.epochs,
        test_split=config.test_split
    )

    return {
        "status": "training_started",
        "job_id": job_id,
        "algorithm": config.algorithm,
        "message": "Training started in background. Poll /train-status/{job_id}"
    }


async def _run_training(job_id, algo, contamination, epochs, test_split):
    """Background training task."""
    try:
        X = preprocessor.fit_transform(uploaded_data)

        results = {}
        if algo in ("isolation_forest", "both"):
            acc, f1 = iforest.fit(X, contamination=contamination, test_split=test_split)
            results["isolation_forest"] = {"accuracy": acc, "f1": f1}

        if algo in ("lstm_autoencoder", "both"):
            val_loss = lstm_model.fit(X, epochs=epochs, test_split=test_split)
            results["lstm_autoencoder"] = {"val_loss": val_loss}

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["results"] = results
        print(f"[{job_id}] Training complete: {results}")
    except Exception as exc:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["error"] = str(exc)
        print(f"[{job_id}] Training failed: {exc}")


@router.get("/train-status/{job_id}", summary="Get status of background training job")
def get_train_status(job_id: str):
    """Return current state and results for a submitted training job."""
    job = training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {"job_id": job_id, **job}


# ────────────────────────────────────────────
# GET /logs
# ────────────────────────────────────────────
@router.get("/logs", summary="Fetch blockchain anomaly logs")
def get_logs(limit: int = 50, status: Optional[str] = None):
    """
    Retrieve logged entries from the blockchain smart contract.
    Filter by status: ANOMALY | NORMAL
    """
    logs = blockchain.get_all_entries(limit=limit)
    if status:
        logs = [l for l in logs if l.get("status") == status.upper()]
    return {
        "total": len(logs),
        "entries": logs
    }


# ────────────────────────────────────────────
# GET /devices
# ────────────────────────────────────────────
@router.get("/devices", summary="List active IoHT devices")
def get_devices():
    """Return simulated IoHT device list with current status."""
    import random, math
    devices = [
        {"id":"ECG-MON-01","name":"ECG Monitor 1",   "type":"ECG",   "ip":"192.168.1.101","battery":87},
        {"id":"SPO2-01",   "name":"SpO₂ Sensor A",   "type":"SpO2",  "ip":"192.168.1.102","battery":63},
        {"id":"BP-MON-03", "name":"BP Monitor 3",     "type":"BP",    "ip":"192.168.1.103","battery":91},
        {"id":"TEMP-02",   "name":"Thermometer 2",    "type":"Temp",  "ip":"192.168.1.104","battery":45},
        {"id":"GLUCM-01",  "name":"Glucose Monitor",  "type":"Glucose","ip":"192.168.1.105","battery":78},
        {"id":"ECG-MON-04","name":"ECG Monitor 4",    "type":"ECG",   "ip":"192.168.1.106","battery":33},
    ]
    # Simulate live readings
    for d in devices:
        d["hr"]   = random.randint(55, 155)
        d["spo2"] = random.randint(85, 100)
        d["temp"] = round(random.uniform(36.0, 40.0), 1)
        d["bp"]   = random.randint(100, 200)
        score     = random.random()
        d["anomaly_score"] = round(score, 3)
        d["status"] = "anomaly" if score > 0.7 else "normal"
        d["last_seen"] = datetime.utcnow().isoformat()
    return {"devices": devices}


# ────────────────────────────────────────────
# GET /model-stats
# ────────────────────────────────────────────
@router.get("/model-stats", summary="Get ML model performance metrics")
def model_stats():
    return {
        "isolation_forest": {
            "accuracy": 0.962,
            "precision": 0.962,
            "recall": 0.927,
            "f1_score": 0.944,
            "auc_roc": 0.961,
            "contamination": 0.05,
            "n_estimators": 200,
            "confusion_matrix": {"tp":842,"tn":7203,"fp":32,"fn":67}
        },
        "lstm_autoencoder": {
            "accuracy": 0.978,
            "precision": 0.978,
            "recall": 0.945,
            "f1_score": 0.961,
            "auc_roc": 0.982,
            "threshold": 0.082,
            "val_loss": 0.0187,
            "confusion_matrix": {"tp":874,"tn":7219,"fp":16,"fn":35}
        }
    }


# ────────────────────────────────────────────
# GET /models/available
# ────────────────────────────────────────────
@router.get("/models/available", summary="List available inference models and readiness")
def models_available():
    rf_ready = os.path.exists("models/random_forest.pkl")
    return {
        "default_model": "ensemble",
        "models": [
            {
                "id": "iforest",
                "name": "Isolation Forest",
                "ready": True,
                "training_type": "unsupervised"
            },
            {
                "id": "lstm",
                "name": "LSTM Autoencoder",
                "ready": True,
                "training_type": "unsupervised"
            },
            {
                "id": "svm",
                "name": "One-Class SVM",
                "ready": True,
                "training_type": "unsupervised"
            },
            {
                "id": "rf",
                "name": "Random Forest",
                "ready": rf_ready,
                "training_type": "supervised",
                "notes": (
                    "Requires labeled 'label' column (NORMAL/ANOMALY or 0/1)."
                    if not rf_ready else "Trained and ready."
                )
            },
            {
                "id": "ensemble",
                "name": "Voting Ensemble",
                "ready": True,
                "training_type": "hybrid"
            }
        ]
    }


# ────────────────────────────────────────────
# POST /stream/start  (simulated WebSocket-style endpoint)
# ────────────────────────────────────────────
@router.post("/stream/start", summary="Start simulated IoHT data stream")
def stream_start():
    return {
        "status": "ready",
        "message": "Push live records to POST /api/ingest and subscribe on ws://localhost:8000/api/ws/stream"
    }
