"""
Two topic-extraction techniques:

  1. LDA -- Latent Dirichlet Allocation (probabilistic), fitted on term counts.
  2. NMF -- Non-negative Matrix Factorisation (linear), fitted on TF-IDF.

Both return human-readable topics (their top terms) plus a simple
topic-diversity score = share of unique words across all topics' top terms.
"""
from sklearn.decomposition import LatentDirichletAllocation, NMF
from config import N_TOPICS, N_TOP_TERMS, RANDOM_STATE


def _top_terms(model, feature_names, n_top=N_TOP_TERMS):
    topics = []
    for weights in model.components_:
        idx = weights.argsort()[::-1][:n_top]
        topics.append([feature_names[i] for i in idx])
    return topics


def fit_lda(count_matrix, feature_names, n_topics=N_TOPICS):
    model = LatentDirichletAllocation(
        n_components=n_topics, learning_method="batch",
        max_iter=25, random_state=RANDOM_STATE)
    model.fit(count_matrix)
    return model, _top_terms(model, feature_names)


def fit_nmf(tfidf_matrix, feature_names, n_topics=N_TOPICS):
    model = NMF(n_components=n_topics, init="nndsvda",
                max_iter=400, random_state=RANDOM_STATE)
    model.fit(tfidf_matrix)
    return model, _top_terms(model, feature_names)


def topic_diversity(topics):
    """Fraction of distinct words among all top-terms (1.0 = no overlap)."""
    all_terms = [t for topic in topics for t in topic]
    return len(set(all_terms)) / len(all_terms) if all_terms else 0.0


def format_topics(topics, title):
    lines = [title, "=" * len(title)]
    for i, terms in enumerate(topics, 1):
        lines.append(f"Topic {i:2d}: " + ", ".join(terms))
    lines.append(f"(topic diversity = {topic_diversity(topics):.2f})")
    return "\n".join(lines)
