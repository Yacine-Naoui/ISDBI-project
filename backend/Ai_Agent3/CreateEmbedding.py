# Ai_Agent2/CreateEmbedding.py
import os
import sys
from agent3 import create_embedding_db

# Ensure file paths are built relative to this script's directory
dir_path = os.path.dirname(os.path.abspath(__file__))

# PDF files located in the same directory as this script
pdf_filenames = ["FAS 4", "FAS 10", "FAS 32"]
# Build absolute paths
pdf_paths = [os.path.join(dir_path, fname) for fname in pdf_filenames]

# Verify each file exists before processing
for path in pdf_paths:
    if not os.path.isfile(path):
        print(
            f"Error loading {os.path.basename(path)}: File path {path} is not a valid file or url"
        )
        sys.exit(1)
    else:
        print(
            f"Loading {os.path.basename(path)} as {os.path.splitext(os.path.basename(path))[0]}..."
        )

# Attempt to create the embedding database
try:
    vector_store = create_embedding_db(pdf_paths)
    print("Embeddings created successfully.")
except Exception as e:
    print(f"Error during embedding creation: {e}")
    sys.exit(1)
