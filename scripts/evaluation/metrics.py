"""
Evaluation Metrics for RAG Retrieval
------------------------------------

This module calculates standard Information Retrieval (IR) metrics.

Metrics:
1. Precision@K
2. Recall@K
3. Hit Rate@K
4. Mean Reciprocal Rank (MRR)
"""

# ============================================================
# Precision@K
# ============================================================

def precision_at_k(retrieved_titles, expected_title, k=5):
    """
    Precision@K = Relevant retrieved documents / K
    """

    retrieved = retrieved_titles[:k]

    relevant = sum(
        1
        for title in retrieved
        if expected_title.lower() in title.lower()
    )

    return relevant / k


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(retrieved_titles, expected_title, k=5):
    """
    Recall@K

    Since there is one expected correct article,
    Recall is either 1 or 0.
    """

    retrieved = retrieved_titles[:k]

    for title in retrieved:
        if expected_title.lower() in title.lower():
            return 1.0

    return 0.0


# ============================================================
# Hit Rate@K
# ============================================================

def hit_rate_at_k(retrieved_titles, expected_title, k=5):
    """
    Returns 1 if expected document exists
    in Top-K otherwise 0.
    """

    retrieved = retrieved_titles[:k]

    for title in retrieved:
        if expected_title.lower() in title.lower():
            return 1

    return 0


# ============================================================
# Mean Reciprocal Rank
# ============================================================

def reciprocal_rank(retrieved_titles, expected_title):
    """
    RR = 1 / rank
    """

    for rank, title in enumerate(retrieved_titles, start=1):

        if expected_title.lower() in title.lower():

            return 1 / rank

    return 0.0


# ============================================================
# Average Metrics
# ============================================================

def average(values):
    """
    Compute average safely.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)