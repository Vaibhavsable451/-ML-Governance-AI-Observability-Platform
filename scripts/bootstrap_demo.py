"""Train the demo models and run the first governance evaluation: python scripts/bootstrap_demo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.governance import service  # noqa: E402

for p in service.bootstrap_demo():
    print(f"{p['model_id']:18s} risk={p['risk']['score']:5.1f} {p['risk']['level']:8s} status={p['status']}")
