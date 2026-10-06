"""
End-to-end pipeline:
    load -> preprocess -> vectorise (x2) -> compare -> topic models (x2) -> outputs
Run from the repository root:  python src/run_analysis.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import preprocess
import vectorize
import topic_models


def load_data():
    """Read the file in chunks, keep only rows that have a narrative, and stop
    once MAX_ROWS of them are collected (so a 3 GB file loads fast)."""
    text_col = config.TEXT_COLUMN
    max_rows = getattr(config, "MAX_ROWS", None)
    collected, total = [], 0
    for chunk in pd.read_csv(config.DATA_PATH, chunksize=50000, low_memory=False):
        if text_col not in chunk.columns:
            raise KeyError(
                f"Column '{text_col}' not found. Available: {list(chunk.columns)}")
        chunk = chunk[chunk[text_col].notna()]
        if len(chunk):
            collected.append(chunk)
            total += len(chunk)
        if max_rows and total >= max_rows:
            break
    df = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=[text_col])
    if max_rows:
        df = df.head(max_rows)
    texts = df[text_col].astype(str).tolist()
    return df, texts


def plot_topics(topics, title, fname):
    n = len(topics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.2 * rows))
    axes = axes.flatten()
    for ax, (i, terms) in zip(axes, enumerate(topics, 1)):
        scores = list(range(len(terms), 0, -1))
        ax.barh(range(len(terms)), scores, color="#3b6ea5")
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Topic {i}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, fname)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print(f"[1/5] Loading data from {config.DATA_PATH}")
    df, texts = load_data()
    print(f"      {len(texts)} documents loaded.")

    print(f"[2/5] Preprocessing (backend: {preprocess.backend_name()})")
    cleaned, tokenised = preprocess.preprocess_corpus(texts)

    print("[3/5] Vectorising (TF-IDF + dense embeddings)")
    tfidf, tfidf_vec = vectorize.build_tfidf(cleaned)
    counts, count_vec = vectorize.build_counts(cleaned)
    dense, dense_name = vectorize.build_embeddings(tokenised, cleaned)
    comparison = vectorize.compare_vectorizations(tfidf, dense, dense_name)
    print(comparison)

    print("[4/5] Extracting topics (LDA + NMF)")
    lda_model, lda_topics = topic_models.fit_lda(counts, count_vec.get_feature_names_out())
    nmf_model, nmf_topics = topic_models.fit_nmf(tfidf, tfidf_vec.get_feature_names_out())
    lda_txt = topic_models.format_topics(lda_topics, "LDA topics")
    nmf_txt = topic_models.format_topics(nmf_topics, "NMF topics")
    print("\n" + lda_txt + "\n\n" + nmf_txt)

    print("[5/5] Writing outputs")
    with open(os.path.join(config.OUTPUT_DIR, "topics.txt"), "w", encoding="utf-8") as f:
        f.write(comparison + "\n\n" + lda_txt + "\n\n" + nmf_txt + "\n")
    fig_lda = plot_topics(lda_topics, "LDA - top terms per topic", "lda_topics.png")
    fig_nmf = plot_topics(nmf_topics, "NMF - top terms per topic", "nmf_topics.png")
    print(f"      Saved: outputs/topics.txt, {os.path.basename(fig_lda)}, "
          f"{os.path.basename(fig_nmf)}")
    print("Done.")


if __name__ == "__main__":
    main()