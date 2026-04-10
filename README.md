# 🛡️ IoHT-Shield
## Blockchain-Assisted Anomaly Detection Framework for Securing IoHT Devices

> **Final Year Project** | IEEE Research Paper Implementation
> Stack: FastAPI · Scikit-learn · TensorFlow/Keras · Ethereum · Solidity · React.js

---

## 📁 Project Structure

```
IoHT-Shield/
├── dashboard.html                  ← Standalone web app (open in browser, no setup needed)
├── backend/
│   ├── main.py                     ← FastAPI application entry point
│   ├── requirements.txt
│   ├── api/
│   │   ├── routes.py               ← All REST API endpoints
│   │   └── schemas.py              ← Pydantic request/response models
│   ├── ml/
│   │   ├── isolation_forest.py     ← Isolation Forest anomaly detector
│   │   ├── lstm_autoencoder.py     ← LSTM Autoencoder for time-series
│   │   └── preprocessor.py        ← Feature engineering & normalisation
│   ├── blockchain/
│   │   ├── web3_client.py          ← Web3.py Ethereum integration
│   │   ├── deploy.py               ← Contract deployment script
│   │   └── contracts/
│   │       └── AnomalyLog.sol      ← Smart contract (Solidity 0.8.19)
│   └── models/                     ← Saved ML models (auto-created)
└── docs/
    └── IEEE_Research_Paper.md      ← Full research paper
```

---

## 🚀 Quick Start (Standalone Demo)

**No setup required!** Open `dashboard.html` in any modern browser:

```
Login: admin / iothshield
```

Features available without backend:
- ✅ Live anomaly monitoring simulation
- ✅ Blockchain log visualisation
- ✅ ML model metrics & ROC curves
- ✅ Predict anomaly from manual input
- ✅ System architecture & code reference

---

## ⚙️ Full System Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (for React frontend, optional)
- Node.js with npm install -g ganache (for blockchain)

---

### Step 1 — Install Backend

```bash
cd backend
pip install -r requirements.txt
```

---

### Step 2 — Start Blockchain (Ganache)

```bash
# Install Ganache CLI globally
npm install -g ganache

# Start local Ethereum testnet
ganache --port 8545 --accounts 10 --deterministic
```

Keep this running in a terminal.

---

### Step 3 — Deploy Smart Contract

```bash
cd backend
python blockchain/deploy.py
```

This will:
- Compile `AnomalyLog.sol`
- Deploy to Ganache
- Save ABI to `blockchain/contracts/AnomalyLog_abi.json`
- Write `CONTRACT_ADDRESS` to `.env`

---

### Step 4 — Run FastAPI Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

### Step 5 — Test the API

```bash
# Health check
curl http://localhost:8000/

# Run a prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "ECG-MON-01",
    "heart_rate": 145,
    "spo2": 87,
    "temperature": 39.1,
    "bp_systolic": 185,
    "model": "ensemble"
  }'

# Get blockchain logs
curl http://localhost:8000/api/logs?limit=10

# Get device list
curl http://localhost:8000/api/devices
```

---

## 📡 API Reference

| Method | Endpoint            | Description                          |
|--------|---------------------|--------------------------------------|
| POST   | `/api/upload-data`  | Upload CSV/JSON IoHT dataset         |
| POST   | `/api/predict`      | Detect anomaly + auto-log blockchain |
| POST   | `/api/train`        | Train/retrain ML models              |
| GET    | `/api/logs`         | Fetch blockchain anomaly logs        |
| GET    | `/api/devices`      | List active IoHT devices             |
| GET    | `/api/model-stats`  | ML model performance metrics         |
| POST   | `/api/stream/start` | Start simulated IoHT data stream     |

---

## 🧠 ML Models

### Isolation Forest
- Unsupervised, no labels required
- Detects anomalies by isolating outliers
- Returns score: negative = more anomalous
- Accuracy: **96.2%** | F1: **0.944** | AUC: **0.961**

### LSTM Autoencoder
- Learns normal temporal patterns
- Flags high reconstruction error as anomaly
- Threshold auto-set at 95th percentile of training errors
- Accuracy: **97.8%** | F1: **0.961** | AUC: **0.982**

### Ensemble
- Decision: anomaly if either model flags
- Combined accuracy: **98.1%**

---

## ⛓️ Blockchain

**Contract:** `AnomalyLog.sol`

Each anomaly event is stored on-chain with:
- `device_id` — IoHT device identifier
- `timestamp` — ISO-8601 detection time
- `anomalyScore` — fused ML score (×10000 integer)
- `ifScore` — Isolation Forest score
- `lstmError` — LSTM reconstruction error
- `status` — "ANOMALY" or "NORMAL"
- `loggedBy` — Ethereum address of logger
- `loggedAt` — Block timestamp

Average gas: ~21,000 per transaction

---

## 📄 Research Paper

Full IEEE-format paper available in `docs/IEEE_Research_Paper.md`

**Results Summary:**
- Isolation Forest: 96.2% accuracy
- LSTM Autoencoder: 97.8% accuracy
- Ensemble: 98.1% accuracy, F1=0.969
- End-to-end latency: <180 ms (including blockchain)

---

## 👤 Author

**Project:** Blockchain-Assisted Anomaly Detection Framework for Securing IoHT Devices
**Type:** Final Year Engineering Project
**Year:** 2025

---

## 📜 License

MIT License — Educational use permitted with attribution.


---

## One-Click Run

From the project root, run:

```bash
python run_app.py
```

This starts the FastAPI backend on http://localhost:8000.
