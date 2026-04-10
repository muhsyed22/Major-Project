"""
blockchain/deploy.py
Deploy AnomalyLog.sol to local Ganache blockchain.
Run: python blockchain/deploy.py
"""

import json, os
from web3 import Web3
from solcx import compile_standard, install_solc

GANACHE_URL   = "http://127.0.0.1:8545"
CONTRACT_FILE = os.path.join(os.path.dirname(__file__), "contracts/AnomalyLog.sol")
ABI_OUT       = os.path.join(os.path.dirname(__file__), "contracts/AnomalyLog_abi.json")
ENV_FILE      = ".env"


def deploy():
    print("─" * 50)
    print("IoHT-Shield · Smart Contract Deployment")
    print("─" * 50)

    # 1. Connect to Ganache
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    assert w3.is_connected(), "❌  Cannot connect to Ganache. Is it running?"
    account = w3.eth.accounts[0]
    print(f"✅  Connected to Ganache | Account: {account}")

    # 2. Install Solidity compiler
    install_solc("0.8.19")
    print("✅  solc 0.8.19 ready")

    # 3. Compile contract
    with open(CONTRACT_FILE) as f:
        source = f.read()

    compiled = compile_standard({
        "language": "Solidity",
        "sources": {"AnomalyLog.sol": {"content": source}},
        "settings": {
            "outputSelection": {
                "*": {"*": ["abi", "evm.bytecode"]}
            }
        }
    }, solc_version="0.8.19")

    contract_data = compiled["contracts"]["AnomalyLog.sol"]["AnomalyLog"]
    abi      = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]
    print("✅  Contract compiled successfully")

    # 4. Deploy
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash  = Contract.constructor().transact({
        "from": account,
        "gas": 3_000_000
    })
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash)
    addr     = receipt["contractAddress"]
    print(f"✅  Deployed at: {addr}")
    print(f"    Gas used: {receipt['gasUsed']:,}")

    # 5. Save ABI
    with open(ABI_OUT, "w") as f:
        json.dump(abi, f, indent=2)
    print(f"✅  ABI saved to {ABI_OUT}")

    # 6. Append to .env
    env_line = f"\nCONTRACT_ADDRESS={addr}\n"
    with open(ENV_FILE, "a") as f:
        f.write(env_line)
    print(f"✅  CONTRACT_ADDRESS written to {ENV_FILE}")
    print("─" * 50)
    print("🚀  Deployment complete! Start the backend:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print("─" * 50)


if __name__ == "__main__":
    deploy()
