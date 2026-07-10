from sentence_transformers import models

print("Loading Transformer...")

transformer = models.Transformer(
    "models/muril_base",
    max_seq_length=128,
)

print("SUCCESS!")
print(transformer.get_embedding_dimension())