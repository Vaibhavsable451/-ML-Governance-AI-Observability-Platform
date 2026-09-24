"""Create the Pinecone index (if missing) and upload the policy documents: python scripts/index_policies.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402
from rag.vectorstore.pinecone_store import sync_policies  # noqa: E402

if not config.PINECONE_API_KEY:
    sys.exit("Set PINECONE_API_KEY in .env first.")
print(f"Indexed {sync_policies()} chunks into '{config.PINECONE_INDEX}' (namespace '{config.PINECONE_NAMESPACE}').")
