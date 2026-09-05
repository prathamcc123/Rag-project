import json
import gc
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate Kannada embeddings using a SentenceTransformer model."
)

parser.add_argument(
    "model_path",
    type=str,
    help="Path to the SentenceTransformer model"
)

args = parser.parse_args()

MODEL_PATH = args.model_path


# ============================================================
# PATHS
# ============================================================

CHUNK_DIR = Path("data/chunks/kannada")

OUTPUT_DIR = Path("data/embeddings/kannada")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

FINAL_EMBEDDING_FILE = OUTPUT_DIR / "embeddings.npy"

FINAL_METADATA_FILE = OUTPUT_DIR / "metadata.json"

METADATA_CHECKPOINT = CHECKPOINT_DIR / "metadata_checkpoint.json"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 4

SAVE_EVERY = 500


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BANNER
# ============================================================

print("=" * 70)
print("KANNADA EMBEDDING GENERATOR")
print("=" * 70)

print(f"Device            : {DEVICE}")
print(f"Model             : {MODEL_PATH}")
print(f"Chunk Directory   : {CHUNK_DIR}")
print(f"Output Directory  : {OUTPUT_DIR}")
print(f"Batch Size        : {BATCH_SIZE}")
print(f"Checkpoint Every  : {SAVE_EVERY} chunks")

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SentenceTransformer model...\n")

model = SentenceTransformer(
    MODEL_PATH,
    device=DEVICE
)

print("✓ Model Loaded Successfully")

print(
    f"Embedding Dimension : "
    f"{model.get_embedding_dimension()}"
)


# ============================================================
# FIND CHUNK FILES
# ============================================================

chunk_files = sorted(
    CHUNK_DIR.glob("*.json")
)

if not chunk_files:

    raise FileNotFoundError(
        f"No chunk files found inside {CHUNK_DIR}"
    )

print(
    f"\nTotal Chunk Files Found : "
    f"{len(chunk_files)}"
)


# ============================================================
# RESUME SUPPORT
# ============================================================

existing_metadata = []

existing_embeddings = []

processed_files = set()


if METADATA_CHECKPOINT.exists():

    print("\nCheckpoint found.")

    with open(
        METADATA_CHECKPOINT,
        "r",
        encoding="utf-8"
    ) as f:

        existing_metadata = json.load(f)

    processed_files = {
        item["chunk_id"]
        for item in existing_metadata
        if item.get("chunk_id")
    }

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob(
            "embeddings_part_*.npy"
        )
    )

    for emb_file in checkpoint_files:

        existing_embeddings.append(
            np.load(emb_file)
        )

    print(
        f"Resuming from "
        f"{len(processed_files)} processed chunks."
    )

else:

    print(
        "\nNo previous checkpoint found."
    )

    print(
        "Starting from scratch."
    )


# ============================================================
# READ CHUNKS
# ============================================================

texts = []

metadata = []

print("\nScanning Kannada chunks...\n")


for file in tqdm(
    chunk_files,
    desc="Reading Chunks"
):

    try:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        # ----------------------------------------------------
        # Chunk files contain a LIST of chunks
        # ----------------------------------------------------

        if isinstance(data, dict):

            chunks = [data]

        elif isinstance(data, list):

            chunks = data

        else:

            print(
                f"\nSkipping invalid file: "
                f"{file.name}"
            )

            continue


        for chunk in chunks:

            if not isinstance(
                chunk,
                dict
            ):
                continue


            chunk_id = chunk.get(
                "chunk_id"
            )

            if not chunk_id:

                continue


            if chunk_id in processed_files:

                continue


            text = chunk.get(
                "content",
                ""
            ).strip()


            if not text:

                continue


            texts.append(text)


            metadata.append(
                {
                    "chunk_id": chunk_id,

                    "title": chunk.get(
                        "title"
                    ),

                    "summary": chunk.get(
                        "summary"
                    ),

                    "language": "kn",

                    "url": chunk.get(
                        "url"
                    ),

                    "chunk_index": chunk.get(
                        "chunk_index"
                    ),

                    "total_chunks": chunk.get(
                        "total_chunks"
                    ),

                    "content": text,

                    "word_count": len(
                        text.split()
                    )
                }
            )


    except Exception as e:

        print(
            f"\nSkipping {file.name}"
        )

        print(e)


print("\n" + "=" * 70)

print(
    f"New Chunks To Process : "
    f"{len(texts)}"
)

print("=" * 70)


# ============================================================
# NOTHING TO PROCESS
# ============================================================

if len(texts) == 0:

    print(
        "\nEverything is already embedded."
    )

    sys.exit(0)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print(
    "\nGenerating Kannada embeddings...\n"
)


total = len(texts)

start = 0

checkpoint_index = len(
    existing_embeddings
)


while start < total:

    end = min(
        start + SAVE_EVERY,
        total
    )


    batch_texts = texts[
        start:end
    ]

    batch_metadata = metadata[
        start:end
    ]


    print(
        f"\nProcessing Chunks "
        f"{start + 1} -> {end}"
    )


    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    with torch.no_grad():

        batch_embeddings = model.encode(

            batch_texts,

            batch_size=BATCH_SIZE,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True

        )


    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    checkpoint_file = (
        CHECKPOINT_DIR
        / f"embeddings_part_"
          f"{checkpoint_index:04d}.npy"
    )


    np.save(
        checkpoint_file,
        batch_embeddings
    )


    existing_embeddings.append(
        batch_embeddings
    )


    existing_metadata.extend(
        batch_metadata
    )


    # --------------------------------------------------------
    # SAVE METADATA CHECKPOINT
    # --------------------------------------------------------

    with open(
        METADATA_CHECKPOINT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            existing_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        f"\n✓ Checkpoint Saved -> "
        f"{checkpoint_file.name}"
    )


    print(
        f"Processed : "
        f"{len(existing_metadata)} chunks"
    )


    checkpoint_index += 1

    start = end


    # --------------------------------------------------------
    # FREE MEMORY
    # --------------------------------------------------------

    del batch_embeddings

    del batch_texts

    del batch_metadata

    gc.collect()


    if DEVICE == "mps":

        try:

            torch.mps.empty_cache()

        except Exception:

            pass


# ============================================================
# EMBEDDING GENERATION FINISHED
# ============================================================

print("\n" + "=" * 70)

print(
    "EMBEDDING GENERATION FINISHED"
)

print("=" * 70)


# ============================================================
# MERGE CHECKPOINTS
# ============================================================

print(
    "\nMerging checkpoint files..."
)


all_embeddings = []


checkpoint_files = sorted(
    CHECKPOINT_DIR.glob(
        "embeddings_part_*.npy"
    )
)


if not checkpoint_files:

    raise RuntimeError(
        "No checkpoint files found."
    )


for file in tqdm(
    checkpoint_files,
    desc="Loading Checkpoints"
):

    emb = np.load(file)

    all_embeddings.append(
        emb
    )


final_embeddings = np.vstack(
    all_embeddings
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nFinal Embedding Shape : "
    f"{final_embeddings.shape}"
)


print(
    "Final Metadata Count  : "
    f"{len(existing_metadata)}"
)


if (
    final_embeddings.shape[0]
    != len(existing_metadata)
):

    raise RuntimeError(
        "Embedding count does not "
        "match metadata count!"
    )


# ============================================================
# SAVE FINAL EMBEDDINGS
# ============================================================

print(
    "\nSaving final embeddings..."
)


np.save(
    FINAL_EMBEDDING_FILE,
    final_embeddings
)


print(
    f"✓ Saved -> "
    f"{FINAL_EMBEDDING_FILE}"
)


# ============================================================
# SAVE FINAL METADATA
# ============================================================

with open(
    FINAL_METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        existing_metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"✓ Saved -> "
    f"{FINAL_METADATA_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("KANNADA EMBEDDINGS COMPLETE")
print("=" * 70)

print(
    f"Chunks Embedded : "
    f"{final_embeddings.shape[0]}"
)

print(
    f"Embedding Size  : "
    f"{final_embeddings.shape[1]}"
)

print(
    f"Embeddings      : "
    f"{FINAL_EMBEDDING_FILE}"
)

print(
    f"Metadata        : "
    f"{FINAL_METADATA_FILE}"
)

print("=" * 70)
