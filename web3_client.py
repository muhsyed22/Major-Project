"""
blockchain/web3_client.py
Web3.py Integration — Connects FastAPI backend to Ethereum (Ganache)
"""

import json, os
from datetime import datetime
from typing import Optional

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("[Blockchain] web3 not installed — using in-memory ledger stub.")

# ── Config
GANACHE_URL    = os.getenv("GANACHE_URL", "http://127.0.0.1:8545")
CONTRACT_FILE  = os.path.join(os.path.dirname(__file__), "contracts/AnomalyLog_abi.json")
CONTRACT_ADDR  = os.getenv("CONTRACT_ADDRESS", "")   # Set after deploy


class BlockchainClient:
    """
    Wraps Web3.py to interact with the deployed AnomalyLog smart contract.
    Falls back to an in-memory ledger if Ganache is unavailable.
    """

    def __init__(self):
        self.w3           = None
        self.contract     = None
        self.account      = None
        self._in_memory   = []       # Fallback ledger
        self._block_num   = 140
        self._connected   = False
        self._connect()

    def _connect(self):
        """Attempt to connect to Ganache."""
        if not WEB3_AVAILABLE:
            print("[Blockchain] Stub: in-memory ledger active.")
            return

        try:
            self.w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
            if not self.w3.is_connected():
                raise ConnectionError("Cannot reach Ganache")

            self.account = self.w3.eth.accounts[0]

            if os.path.exists(CONTRACT_FILE) and CONTRACT_ADDR:
                with open(CONTRACT_FILE) as f:
                    abi = json.load(f)
                self.contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(CONTRACT_ADDR),
                    abi=abi
                )
                self._connected = True
                print(f"[Blockchain] Connected to Ganache at {GANACHE_URL}")
            else:
                print("[Blockchain] Contract ABI/address missing — stub mode.")
        except Exception as e:
            print(f"[Blockchain] Connection failed: {e} — stub mode.")

    def log_entry(
        self,
        device_id:     str,
        timestamp:     str,
        anomaly_score: float,
        if_score:      float,
        lstm_error:    float,
        status:        str
    ) -> str:
        """
        Write an anomaly entry to the blockchain.
        Returns the transaction hash (or a simulated hash in stub mode).
        """
        self._block_num += 1

        if self._connected and self.contract:
            try:
                tx_hash = self.contract.functions.logEntry(
                    device_id,
                    timestamp,
                    int(anomaly_score * 10000),  # Store as integer (×10000)
                    int(if_score * 10000),
                    int(lstm_error * 10000),
                    status
                ).transact({"from": self.account, "gas": 200_000})
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                tx_str  = receipt["transactionHash"].hex()
                gas     = receipt["gasUsed"]
            except Exception as e:
                print(f"[Blockchain] Transaction failed: {e} — using stub.")
                tx_str, gas = self._stub_tx()
        else:
            tx_str, gas = self._stub_tx()

        # Always record in memory for fast API reads
        entry = {
            "block_number":  self._block_num,
            "tx_hash":       tx_str,
            "device_id":     device_id,
            "timestamp":     timestamp,
            "anomaly_score": anomaly_score,
            "if_score":      if_score,
            "lstm_error":    lstm_error,
            "status":        status,
            "gas_used":      gas
        }
        self._in_memory.insert(0, entry)   # Newest first
        print(f"[Blockchain] Logged block #{self._block_num} — {status} — {tx_str[:18]}…")
        return tx_str

    def get_all_entries(self, limit: int = 50) -> list:
        """
        Return logged entries. If connected reads from contract,
        otherwise returns in-memory ledger + seeded demo data.
        """
        if self._connected and self.contract:
            try:
                total = self.contract.functions.getTotalEntries().call()
                entries = []
                for i in range(max(0, total - limit), total):
                    e = self.contract.functions.getEntry(i).call()
                    entries.insert(0, {
                        "block_number":  e[0],
                        "tx_hash":       e[1],
                        "device_id":     e[2],
                        "timestamp":     e[3],
                        "anomaly_score": e[4] / 10000,
                        "if_score":      e[5] / 10000,
                        "lstm_error":    e[6] / 10000,
                        "status":        e[7],
                        "gas_used":      21000
                    })
                return entries
            except Exception:
                pass

        # Return stub data
        seed = self._seed_demo_entries()
        combined = self._in_memory + seed
        return combined[:limit]

    def _stub_tx(self):
        """Generate a realistic-looking fake transaction hash."""
        import random, string
        h = "0x" + "".join(random.choices("0123456789abcdef", k=40))
        return h, random.randint(20900, 21600)

    def _seed_demo_entries(self) -> list:
        """Pre-seeded demo blockchain entries for the dashboard."""
        import random
        devices  = ["ECG-MON-01","SPO2-01","BP-MON-03","TEMP-02","GLUCM-01","ECG-MON-04","PULSE-02","INFPM-01"]
        statuses = ["ANOMALY","ANOMALY","ANOMALY","NORMAL","NORMAL","NORMAL","NORMAL","NORMAL"]
        entries  = []
        for i in range(30):
            is_anom = statuses[i % len(statuses)] == "ANOMALY"
            entry = {
                "block_number":  140 - i,
                "tx_hash":       "0x" + "".join(random.choices("0123456789abcdef", k=40)),
                "device_id":     devices[i % len(devices)],
                "timestamp":     f"2025-03-29 {14-i//60:02d}:{59-(i*2)%60:02d}:{random.randint(0,59):02d}",
                "anomaly_score": round(random.uniform(0.75,0.97) if is_anom else random.uniform(0.01,0.15), 4),
                "if_score":      round(-random.uniform(0.6,0.95) if is_anom else random.uniform(0.05,0.25), 4),
                "lstm_error":    round(random.uniform(0.1,0.25) if is_anom else random.uniform(0.01,0.06), 4),
                "status":        "ANOMALY" if is_anom else "NORMAL",
                "gas_used":      random.randint(20900, 21600)
            }
            entries.append(entry)
        return entries
