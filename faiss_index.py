import faiss
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Define file paths
index_file = "data/faiss_index.bin"
dataset_paths = [
    "data\Combined Data.csv",
    "data\mental_health_data final data.csv",
    "data\data.csv"
]

# Step 1: Load all datasets
print("🔄 Step 1: Loading datasets...")
statements = []  # Initialize empty list for statements

for dataset_path in dataset_paths:
    if os.path.exists(dataset_path):  # Ensure the file exists
        df = pd.read_csv(dataset_path, dtype=str)  # Load all columns as strings
        text_column = df.columns[0]  # Assuming first column has text
        df[text_column] = df[text_column].astype(str)  # Convert column to string
        statements.extend(df[text_column].dropna().tolist())  # Append non-null statements
        print(f"✅ Loaded {len(df)} rows from {dataset_path}")
    else:
        print(f"⚠️ Warning: {dataset_path} not found!")

print(f"📌 Step 2: Combined dataset has {len(statements)} total statements.")

# Step 3: Load Sentence Transformer model
print("🔄 Step 3: Loading Sentence Transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

# Step 4: Generate vector embeddings
print("🔄 Step 4: Generating embeddings...")
statements = [str(s) for s in statements]  # Ensure all values are strings
statement_embeddings = np.array(embedder.encode(statements, convert_to_tensor=True))
dimension = statement_embeddings.shape[1]
print(f"✅ Embeddings generated! Shape: {statement_embeddings.shape}")

# Step 5: Initialize FAISS index
print("🔄 Step 5: Creating FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(statement_embeddings)
print("✅ FAISS index created and embeddings added!")

# Step 6: Save FAISS index
faiss.write_index(index, index_file)
print(f"✅ FAISS index saved to {index_file} successfully!")


# import faiss
# import os
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# # Define paths
# index_file = "data/faiss_index.bin"
# dataset_path = "data/combined_data.csv"

# print("🔄 Step 1: Loading dataset...")

# # Load dataset
# df = pd.read_csv(dataset_path)
# statements = df["statement"].fillna("").tolist()

# print(f"📌 Step 2: Dataset loaded with {len(statements)} statements.")

# # Load Sentence Transformer model for embeddings
# print("🔄 Step 3: Loading Sentence Transformer model...")
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# print("✅ Model loaded successfully!")

# # Convert text statements into vector embeddings
# print("🔄 Step 4: Generating embeddings...")
# statement_embeddings = np.array(embedder.encode(statements, convert_to_tensor=True))
# dimension = statement_embeddings.shape[1]
# print("✅ Embeddings generated!")

# def save_faiss_index(index):
#     """Save FAISS index to disk."""
#     faiss.write_index(index, index_file)
#     print("✅ FAISS index saved successfully!")

# def load_faiss_index():
#     """Load FAISS index from disk if available."""
#     if os.path.exists(index_file):
#         print("🔄 Loading existing FAISS index...")
#         return faiss.read_index(index_file)
#     print("⚠️ No existing FAISS index found. Creating a new one.")
#     return None

# # Load existing FAISS index or create a new one
# print("🔄 Step 5: Checking for existing FAISS index...")
# index = load_faiss_index()

# if index is None:
#     print("📌 Step 6: Creating a new FAISS index...")
#     index = faiss.IndexFlatL2(dimension)
#     print("✅ FAISS index initialized!")

#     print("🔄 Step 7: Adding embeddings to index...")
#     index.add(statement_embeddings)
#     print("✅ Embeddings added to FAISS index!")

#     save_faiss_index(index)
#     print("✅ FAISS Index created and stored successfully!")
# else:
#     print("✅ FAISS Index loaded successfully!")