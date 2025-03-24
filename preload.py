from sentence_transformers import SentenceTransformer
print("Starting the Model!")
model = SentenceTransformer("all-MiniLM-L6-v2")  # This forces model download
print("Model is ready!")
