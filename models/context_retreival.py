import pandas as pd
import numpy as np
import faiss
import os
import time
from sentence_transformers import SentenceTransformer

# 📌 Checking if FAISS index exists before loading
print("📌 Checking if FAISS index exists...")

if os.path.exists("data/faiss_index.bin"):
    print("✅ FAISS index found! Loading now...")
else:
    print("❌ FAISS index NOT found! The chatbot might be recomputing embeddings.")

# Measure FAISS loading time
start_time = time.time()
index = faiss.read_index("data/faiss_index.bin")
print(f"✅ FAISS Index loaded in {time.time() - start_time:.2f} seconds!")

# Load datasets
print("📌 Loading datasets...")
combined_data = pd.read_csv("data\Combined Data.csv")
mental_health_data = pd.read_csv("data\mental_health_data final data.csv")
user_data = pd.read_csv("data\data.csv")
print("✅ Datasets loaded successfully!")

# Initialize Sentence Transformer model for embeddings
print("📌 Loading Sentence Transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

# Extract statements for FAISS retrieval
statements = combined_data["statement"].fillna("").tolist()

# Store index mapping for retrieval
statement_index = {i: statements[i] for i in range(len(statements))}

# def retrieve_relevant_context(user_input, top_k=3):
#     """Retrieves relevant mental health-related statements from dataset using FAISS."""
#     print(f"🔎 Searching FAISS index for relevant context on: {user_input}")
#     query_embedding = np.array(embedder.encode([user_input], convert_to_tensor=True))
#     _, indices = index.search(query_embedding, top_k)
#     retrieved_statements = [statement_index[i] for i in indices[0]]
#     print("✅ Retrieved relevant statements from FAISS.")
#     return " ".join(retrieved_statements)
def retrieve_relevant_context(user_input, top_k=2):  # Reduced to top 2 results
    """Retrieves the most relevant mental health-related statements using FAISS."""
    print(f"🔎 Searching FAISS index for: {user_input}")
    query_embedding = np.array(embedder.encode([user_input], convert_to_tensor=True))
    _, indices = index.search(query_embedding, top_k)

    retrieved_statements = [statement_index.get(int(i), "") for i in indices[0] if i in statement_index]
    
    # Ensure retrieved context is concise
    retrieved_text = " ".join(retrieved_statements)[:200]  # Max 200 characters
    print(f"✅ Retrieved: {retrieved_text}")

    return retrieved_text if retrieved_text.strip() else "No relevant context found."



def get_stress_related_advice():
    """Suggests advice based on average stress levels in dataset."""
    print("📌 Analyzing stress levels from dataset...")
    avg_stress = mental_health_data["Stress_Level"].value_counts().idxmax()
    
    if avg_stress == "High":
        return "Based on mental health data, stress levels are generally high. Consider relaxation techniques like meditation and deep breathing."
    elif avg_stress == "Medium":
        return "Many individuals report moderate stress. Taking breaks and engaging in physical activities may help."
    else:
        return "Stress levels are generally low, which is great! Keep maintaining a balanced lifestyle."

def get_resource_suggestions():
    """Suggests mental health resources based on user demographics."""
    print("📌 Fetching most common mental health issues by region...")
    common_region = user_data["Region"].value_counts().idxmax()
    common_depression_status = user_data["Dep"].value_counts().idxmax()

    if common_depression_status == "Yes":
        return f"Many users from {common_region} report experiencing depression. Seeking support groups or counseling services in your region may be beneficial."
    else:
        return f"Users from {common_region} generally report stable mental health. Staying connected with friends and family can further support well-being."




# # context_retrieval.py
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# # Load datasets
# combined_data = pd.read_csv("data/combined_data.csv")
# mental_health_data = pd.read_csv("data\mental_health_data_final _data.csv")
# user_data = pd.read_csv("data/data.csv")

# # Initialize Sentence Transformer model for embeddings
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # Convert anxiety statements into embeddings
# statements = combined_data["statement"].fillna("").tolist()
# statement_embeddings = np.array(embedder.encode(statements, convert_to_tensor=True))

# # Initialize FAISS index for similarity search
# dimension = statement_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(statement_embeddings)

# # Store index mapping
# statement_index = {i: statements[i] for i in range(len(statements))}

# def retrieve_relevant_context(user_input, top_k=3):
#     """Retrieves relevant mental health-related statements from dataset."""
#     query_embedding = np.array(embedder.encode([user_input], convert_to_tensor=True))
#     _, indices = index.search(query_embedding, top_k)
#     retrieved_statements = [statement_index[i] for i in indices[0]]
#     return " ".join(retrieved_statements)

# def get_stress_related_advice():
#     """Suggests advice based on average stress levels in dataset."""
#     avg_stress = mental_health_data["Stress_Level"].value_counts().idxmax()
    
#     if avg_stress == "High":
#         return "Based on mental health data, stress levels are generally high. Consider relaxation techniques like meditation and deep breathing."
#     elif avg_stress == "Medium":
#         return "Many individuals report moderate stress. Taking breaks and engaging in physical activities may help."
#     else:
#         return "Stress levels are generally low, which is great! Keep maintaining a balanced lifestyle."

# def get_resource_suggestions():
#     """Suggests mental health resources based on user demographics."""
#     common_region = user_data["Region"].value_counts().idxmax()
#     common_depression_status = user_data["Dep"].value_counts().idxmax()

#     if common_depression_status == "Yes":
#         return f"Many users from {common_region} report experiencing depression. Seeking support groups or counseling services in your region may be beneficial."
#     else:
#         return f"Users from {common_region} generally report stable mental health. Staying connected with friends and family can further support well-being."