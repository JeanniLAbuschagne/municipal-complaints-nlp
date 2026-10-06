"""
Two ways of turning clean text into numeric vectors:

  1. TF-IDF  -- sparse, count/frequency based (scikit-learn).
  2. Dense embeddings -- Word2Vec averaged document vectors (gensim).
     If gensim is unavailable, falls back to Truncated SVD (LSA) over the
     TF-IDF matrix, which also yields dense semantic document vectors.

A small comparison helper reports the practical differences between them.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from config import MIN_DF, MAX_DF, NGRAM_RANGE, RANDOM_STATE


def build_tfidf(cleaned_docs):
    """Sparse TF-IDF representation."""
    vec = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, ngram_range=NGRAM_RANGE)
    matrix = vec.fit_transform(cleaned_docs)
    return matrix, vec


def build_counts(cleaned_docs):
    """Raw term counts -- the natural input for LDA."""
    vec = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF, ngram_range=NGRAM_RANGE)
    matrix = vec.fit_transform(cleaned_docs)
    return matrix, vec


def build_embeddings(tokenised_docs, cleaned_docs, dim=100):
    """
    Dense document vectors.

    Preferred: average of Word2Vec word vectors trained on the corpus (gensim).
    Fallback : Truncated SVD (LSA) over TF-IDF -> dense topic-space vectors.
    Returns (matrix, method_name).
    """
    try:
        from gensim.models import Word2Vec
        w2v = Word2Vec(sentences=tokenised_docs, vector_size=dim, window=5,
                       min_count=MIN_DF, workers=2, seed=RANDOM_STATE, epochs=40)
        vectors = []
        for toks in tokenised_docs:
            vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
            vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))
        return np.vstack(vectors), "Word2Vec (averaged, gensim)"
    except Exception:
        from sklearn.decomposition import TruncatedSVD
        tfidf, _ = build_tfidf(cleaned_docs)
        n_comp = min(dim, tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        dense = svd.fit_transform(tfidf)
        return dense, f"LSA / Truncated SVD ({n_comp} dims, fallback)"


def compare_vectorizations(tfidf_matrix, dense_matrix, dense_name):
    """Return a short human-readable comparison of the two representations."""
    sparsity = 100.0 * (1.0 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    lines = [
        "Vectorisation comparison",
        "-" * 40,
        f"TF-IDF        : shape {tfidf_matrix.shape}, "
        f"{sparsity:.2f}% zeros (sparse, one dimension per term).",
        f"{dense_name:<14}: shape {tuple(dense_matrix.shape)}, "
        f"dense (continuous semantic dimensions).",
        "",
        "TF-IDF is high-dimensional, sparse and directly interpretable per term;",
        "the dense embedding is compact and groups semantically related wording,",
        "but its individual dimensions are not human-readable.",
    ]
    return "\n".join(lines)
