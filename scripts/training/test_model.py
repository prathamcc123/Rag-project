from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/final_model")

print("✅ Model loaded successfully!")

embedding = model.encode("How to grow rice?")
print("Embedding shape:", embedding.shape)