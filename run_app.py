"""Launch the IoHT-Shield FastAPI app."""
import subprocess
import sys

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
    raise SystemExit(subprocess.call(cmd))
