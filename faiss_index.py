import faiss
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Define paths
index_file = "data/faiss_index.bin"
dataset_path = "data/combined_data.csv"

print("🔄 Step 1: Loading dataset...")

# Load dataset
df = pd.read_csv(dataset_path)
statements = df["statement"].fillna("").tolist()

print(f"📌 Step 2: Dataset loaded with {len(statements)} statements.")

# Load Sentence Transformer model for embeddings
print("🔄 Step 3: Loading Sentence Transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

# Convert text statements into vector embeddings
print("🔄 Step 4: Generating embeddings...")
statement_embeddings = np.array(embedder.encode(statements, convert_to_tensor=True))
dimension = statement_embeddings.shape[1]
print("✅ Embeddings generated!")

def save_faiss_index(index):
    """Save FAISS index to disk."""
    faiss.write_index(index, index_file)
    print("✅ FAISS index saved successfully!")

def load_faiss_index():
    """Load FAISS index from disk if available."""
    if os.path.exists(index_file):
        print("🔄 Loading existing FAISS index...")
        return faiss.read_index(index_file)
    print("⚠️ No existing FAISS index found. Creating a new one.")
    return None

# Load existing FAISS index or create a new one
print("🔄 Step 5: Checking for existing FAISS index...")
index = load_faiss_index()

if index is None:
    print("📌 Step 6: Creating a new FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    print("✅ FAISS index initialized!")

    print("🔄 Step 7: Adding embeddings to index...")
    index.add(statement_embeddings)
    print("✅ Embeddings added to FAISS index!")

    save_faiss_index(index)
    print("✅ FAISS Index created and stored successfully!")
else:
    print("✅ FAISS Index loaded successfully!")