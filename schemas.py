"""
api/schemas.py — Pydantic request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    device_id:   str   = Field(..., example="ECG-MON-01")
    heart_rate:  float = Field(..., ge=0, le=300, example=72.0)
    spo2:        float = Field(..., ge=50, le=100, example=98.0)
    temperature: float = Field(..., example=36.5)
    bp_systolic: float = Field(..., example=120.0)
    model:       str   = Field(default="ensemble", example="ensemble")

    class Config:
        schema_extra = {
            "example": {
                "device_id": "ECG-MON-01",
                "heart_rate": 72,
                "spo2": 98,
                "temperature": 36.5,
                "bp_systolic": 120,
                "model": "ensemble"
            }
        }


class PredictResponse(BaseModel):
    device_id:                   str
    timestamp:                   str
    status:                      str
    anomaly_score:               float
    if_score:                    float
    lstm_reconstruction_error:   float
    is_anomaly:                  bool
    blockchain_tx_hash:          Optional[str]
    message:                     str


class TrainRequest(BaseModel):
    algorithm:     str   = Field(default="both", example="both")
    contamination: float = Field(default=0.05, ge=0.01, le=0.5)
    epochs:        int   = Field(default=50, ge=5, le=500)
    test_split:    float = Field(default=0.2, ge=0.1, le=0.4)


class TrainResponse(BaseModel):
    status:    str
    job_id:    str
    algorithm: str
    message:   str


class DeviceData(BaseModel):
    device_id:   str
    heart_rate:  float
    spo2:        float
    temperature: float
    bp_systolic: float
    timestamp:   Optional[str] = None


class LogEntry(BaseModel):
    block_number:  int
    tx_hash:       str
    device_id:     str
    timestamp:     str
    anomaly_score: float
    if_score:      float
    lstm_error:    float
    status:        str
    gas_used:      int


class StreamConfig(BaseModel):
    interval_seconds: float = 2.0
    devices:          list  = []
