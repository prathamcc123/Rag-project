import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    InputExample,
    losses,
    models,
)
from sentence_transformers.training_args import BatchSamplers
from transformers import set_seed

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "models/muril_base"

TRAIN_FILE = "data/training/train.jsonl"
VALID_FILE = "data/training/valid.jsonl"

OUTPUT_DIR = "models/muril_agriculture"

MAX_SEQ_LENGTH = 128

BATCH_SIZE = 2

NUM_EPOCHS = 1

LEARNING_RATE = 2e-5

WARMUP_RATIO = 0.1

SEED = 42

# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("MuRIL Fine-Tuning")
print("=" * 60)
print(f"Device : {device}")
print(f"Model  : {MODEL_NAME}")
print("=" * 60)

# ============================================================
# Load MuRIL
# ============================================================

print("\nLoading MuRIL...")

transformer = models.Transformer(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
)

pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
)

model = SentenceTransformer(
    modules=[transformer, pooling],
    device=device,
)

print("✓ MuRIL loaded successfully")
print(f"Embedding Dimension : {model.get_sentence_embedding_dimension()}")
# ============================================================
# Load JSONL Dataset
# ============================================================

def load_jsonl(path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            examples.append(
    InputExample(
        texts=[
            row["sentence1"],
            row["sentence2"],
        ]
    )
)

    return examples


print("\nLoading training data...")

train_examples = load_jsonl(TRAIN_FILE)
valid_examples = load_jsonl(VALID_FILE)

print(f"Training pairs   : {len(train_examples)}")
print(f"Validation pairs : {len(valid_examples)}")

# ============================================================
# Convert to HuggingFace Dataset
# ============================================================

train_dataset = Dataset.from_dict(
    {
        "sentence1": [x.texts[0] for x in train_examples],
        "sentence2": [x.texts[1] for x in train_examples],
    }
)

valid_dataset = Dataset.from_dict(
    {
        "sentence1": [x.texts[0] for x in valid_examples],
        "sentence2": [x.texts[1] for x in valid_examples],
    }
)

print("\nDataset loaded successfully.")

# ============================================================
# Loss Function
# ============================================================

train_loss = losses.MultipleNegativesRankingLoss(model)

print("Loss Function : MultipleNegativesRankingLoss")

# ============================================================
# Training Arguments
# ============================================================

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    warmup_ratio=WARMUP_RATIO,

    batch_sampler=BatchSamplers.NO_DUPLICATES,

    save_strategy="epoch",

    eval_strategy="epoch",

    logging_steps=50,

    save_total_limit=2,

    load_best_model_at_end=True,

    fp16=False,

    bf16=False,

    seed=SEED,

    report_to="none",
)
# ============================================================
# Trainer
# ============================================================

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    loss=train_loss,
)

# ============================================================
# Train
# ============================================================

print("\n" + "=" * 60)
print("Starting Fine-Tuning...")
print("=" * 60)

trainer.train()

print("\n✓ Training completed successfully.")

# ============================================================
# Save Model
# ============================================================

print("\nSaving model...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

model.save(OUTPUT_DIR)

print(f"✓ Model saved to: {OUTPUT_DIR}")

# ============================================================
# Test Embeddings
# ============================================================

print("\nTesting fine-tuned model...")

sample_sentences = [
    "गेहूं की खेती कैसे करें",
    "धान में कीट नियंत्रण",
    "टमाटर की खेती",
]

embeddings = model.encode(
    sample_sentences,
    convert_to_tensor=True,
)

print(f"Generated embeddings shape: {embeddings.shape}")

print("\n" + "=" * 60)
print("FINE-TUNING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Model Location : {OUTPUT_DIR}")
print(f"Embedding Size : {embeddings.shape[1]}")
print("=" * 60)