import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from Ai_Agent2.agent2 import create_embedding_db

pdf_paths = ["FAS4.pdf", "FAS7.pdf", "FAS10.pdf", "FAS28.pdf", "FAS32.pdf"]
vectore_store = create_embedding_db(pdf_paths)
