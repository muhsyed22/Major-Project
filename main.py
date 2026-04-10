"""
IoHT-Shield: Blockchain-Assisted Anomaly Detection Framework
FastAPI Backend — main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(
    title="IoHT-Shield API",
    description="Blockchain-Assisted Anomaly Detection for IoHT Devices",
    version="2.4.1"
)

# ── CORS (allow React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API router
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {
        "project": "IoHT-Shield",
        "version": "2.4.1",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
