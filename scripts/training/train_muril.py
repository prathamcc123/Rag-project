import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sentence_transformers import SentenceTransformer

from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)

from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
)

# ==========================================================
# CONFIG
# ==========================================================

# Continue training from your existing fine-tuned model
MODEL_NAME = "models/muril_agriculture"

# Dataset
TRAIN_FILE = "data/splits/train.csv"
VALID_FILE = "data/splits/valid.csv"

# Output
OUTPUT_DIR = "models/muril_agriculture_v2"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
LOGGING_DIR = "logs"

# ==========================================================
# TRAINING CONFIG (Optimized for MacBook Air M1 - 8GB)
# ==========================================================

SEED = 42

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

MAX_SEQ_LENGTH = 128

SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# DEVICE
# ==========================================================

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

logger.info(f"Using device: {device}")

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGGING_DIR).mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

logger.info("Loading training dataset...")

train_df = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8-sig",
)

valid_df = pd.read_csv(
    VALID_FILE,
    encoding="utf-8-sig",
)

logger.info(f"Training pairs   : {len(train_df):,}")
logger.info(f"Validation pairs : {len(valid_df):,}")

# ==========================================================
# VERIFY REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "question",
    "positive_text",
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Column '{column}' not found in train.csv"
        )

    if column not in valid_df.columns:
        raise ValueError(
            f"Column '{column}' not found in valid.csv"
        )

# ==========================================================
# KEEP REQUIRED COLUMNS
# ==========================================================

train_df = train_df[
    [
        "question",
        "positive_text",
    ]
].copy()

valid_df = valid_df[
    [
        "question",
        "positive_text",
    ]
].copy()

# ==========================================================
# RENAME COLUMNS
# ==========================================================

train_df.rename(
    columns={
        "question": "anchor",
        "positive_text": "positive",
    },
    inplace=True,
)

valid_df.rename(
    columns={
        "question": "anchor",
        "positive_text": "positive",
    },
    inplace=True,
)

# ==========================================================
# REMOVE NULLS
# ==========================================================

train_df.dropna(inplace=True)
valid_df.dropna(inplace=True)

train_df.reset_index(
    drop=True,
    inplace=True,
)

valid_df.reset_index(
    drop=True,
    inplace=True,
)

logger.info(f"Training after cleaning   : {len(train_df):,}")
logger.info(f"Validation after cleaning : {len(valid_df):,}")

# ==========================================================
# CONVERT TO HUGGINGFACE DATASETS
# ==========================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False,
)

valid_dataset = Dataset.from_pandas(
    valid_df,
    preserve_index=False,
)

logger.info(train_dataset)
logger.info(valid_dataset)

# ==========================================================
# LOAD MuRIL MODEL
# ==========================================================

logger.info("Loading fine-tuned MuRIL model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

model.max_seq_length = MAX_SEQ_LENGTH

logger.info("Model loaded successfully.")

# ==========================================================
# LOSS FUNCTION
# ==========================================================

loss = MultipleNegativesRankingLoss(model)

logger.info("MultipleNegativesRankingLoss initialized.")

print("=" * 60)
print("Everything Loaded Successfully")
print("=" * 60)
print(f"Device             : {device}")
print(f"Model              : {MODEL_NAME}")
print(f"Training pairs     : {len(train_dataset):,}")
print(f"Validation pairs   : {len(valid_dataset):,}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print("=" * 60)
# ==========================================================
# PART 2
# ==========================================================

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

# ==========================================================
# BUILD VALIDATION CORPUS
# ==========================================================

logger.info("Preparing validation evaluator...")

queries = {}
corpus = {}
relevant_docs = {}

for idx, row in valid_df.iterrows():

    query_id = f"q{idx}"
    doc_id = f"d{idx}"

    queries[query_id] = row["anchor"]
    corpus[doc_id] = row["positive"]

    relevant_docs[query_id] = {doc_id}

logger.info(f"Queries : {len(queries):,}")
logger.info(f"Corpus  : {len(corpus):,}")

# ==========================================================
# INFORMATION RETRIEVAL EVALUATOR
# ==========================================================

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="validation",
)

logger.info("Validation evaluator created.")

# ==========================================================
# TRAINING ARGUMENTS
# ==========================================================

training_args = SentenceTransformerTrainingArguments(

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    output_dir=OUTPUT_DIR,

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO,

    # ------------------------------------------------------
    # Mixed Precision
    # ------------------------------------------------------

    # CUDA only
    fp16=torch.cuda.is_available(),
    bf16=False,

    # ------------------------------------------------------
    # Multiple Negatives Ranking Loss
    # ------------------------------------------------------

    batch_sampler=BatchSamplers.NO_DUPLICATES,

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    # ------------------------------------------------------
    # Saving
    # ------------------------------------------------------

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,

    load_best_model_at_end=True,

    metric_for_best_model="eval_validation_cosine_mrr@10",
    greater_is_better=True,

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    logging_dir=LOGGING_DIR,
    report_to="tensorboard",

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    seed=SEED,

    # ------------------------------------------------------
    # MacBook Air M1 Optimization
    # ------------------------------------------------------

    dataloader_num_workers=0,
    dataloader_pin_memory=False,

    remove_unused_columns=False,

    run_name="muril_agriculture_v2",
)

logger.info("Training arguments created.")

# ==========================================================
# TRAINER
# ==========================================================

trainer = SentenceTransformerTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=valid_dataset,

    loss=loss,

    evaluator=ir_evaluator,

)

logger.info("Trainer initialized.")

# ==========================================================
# RESUME FROM CHECKPOINT
# ==========================================================

resume_checkpoint = None

checkpoints = sorted(
    Path(OUTPUT_DIR).glob("checkpoint-*"),
    key=lambda x: int(x.name.split("-")[-1]),
)

if len(checkpoints) > 0:

    resume_checkpoint = str(checkpoints[-1])

    logger.info(
        f"Resuming from checkpoint: {resume_checkpoint}"
    )

else:

    logger.info(
        "No checkpoint found. Starting fresh."
    )
    # ==========================================================
# PART 3
# ==========================================================

# ==========================================================
# TRAIN MODEL
# ==========================================================

logger.info("=" * 60)
logger.info("Starting MuRIL Training...")
logger.info("=" * 60)

trainer.train(
    resume_from_checkpoint=resume_checkpoint
)

logger.info("=" * 60)
logger.info("Training Completed Successfully")
logger.info("=" * 60)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

FINAL_MODEL_DIR = f"{OUTPUT_DIR}/final"

Path(FINAL_MODEL_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(FINAL_MODEL_DIR)

logger.info(f"Model saved to: {FINAL_MODEL_DIR}")

# Save complete SentenceTransformer model
model.save(FINAL_MODEL_DIR)

logger.info("SentenceTransformer model exported successfully.")

# ==========================================================
# FINAL VALIDATION
# ==========================================================

logger.info("=" * 60)
logger.info("Running Final Validation...")
logger.info("=" * 60)

results = ir_evaluator(model)

print("\n")
print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

for metric, value in results.items():

    if isinstance(value, float):
        print(f"{metric:<45} {value:.6f}")
    else:
        print(f"{metric:<45} {value}")

print("=" * 70)

# ==========================================================
# SAVE VALIDATION METRICS
# ==========================================================

metrics_file = Path(OUTPUT_DIR) / "validation_metrics.txt"

with open(metrics_file, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("FINAL VALIDATION RESULTS\n")
    f.write("=" * 70 + "\n")

    for metric, value in results.items():
        f.write(f"{metric}: {value}\n")

logger.info(f"Validation metrics saved to {metrics_file}")

# ==========================================================
# SAVE TRAINING CONFIG
# ==========================================================

config_file = Path(OUTPUT_DIR) / "training_config.txt"

with open(config_file, "w", encoding="utf-8") as f:

    f.write("MuRIL Fine-tuning Configuration\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Base Model          : {MODEL_NAME}\n")
    f.write(f"Training File       : {TRAIN_FILE}\n")
    f.write(f"Validation File     : {VALID_FILE}\n")
    f.write(f"Epochs              : {NUM_EPOCHS}\n")
    f.write(f"Train Batch Size    : {TRAIN_BATCH_SIZE}\n")
    f.write(f"Eval Batch Size     : {EVAL_BATCH_SIZE}\n")
    f.write(f"Learning Rate       : {LEARNING_RATE}\n")
    f.write(f"Weight Decay        : {WEIGHT_DECAY}\n")
    f.write(f"Warmup Ratio        : {WARMUP_RATIO}\n")
    f.write(f"Max Sequence Length : {MAX_SEQ_LENGTH}\n")
    f.write(f"Device              : {device}\n")

logger.info(f"Training configuration saved to {config_file}")

# ==========================================================
# TRAINING SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"Device               : {device}")
print(f"Training pairs       : {len(train_dataset):,}")
print(f"Validation pairs     : {len(valid_dataset):,}")
print(f"Epochs               : {NUM_EPOCHS}")
print(f"Train Batch Size     : {TRAIN_BATCH_SIZE}")
print(f"Eval Batch Size      : {EVAL_BATCH_SIZE}")
print(f"Learning Rate        : {LEARNING_RATE}")
print(f"Max Sequence Length  : {MAX_SEQ_LENGTH}")

print("\nSaved Model:")
print(FINAL_MODEL_DIR)

print("\nValidation Metrics:")

for metric, value in results.items():

    if isinstance(value, float):
        print(f"{metric:<45} {value:.6f}")

print("=" * 70)

logger.info("Training pipeline completed successfully.")