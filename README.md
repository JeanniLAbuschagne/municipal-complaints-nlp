# Municipal Complaints  NLP Topic Extraction

Extracts the most frequently addressed topics from a collection of unstructured
complaint texts, so municipal decision-makers can see the most pressing issues
without reading every complaint by hand.

The pipeline: **load → preprocess → vectorise (two ways) → extract topics (two
ways) → report**.

| Stage | Technique(s) | Library |
|-------|--------------|---------|
| Preprocessing | clean, tokenise, stop-word removal, lemmatise | NLTK (sklearn fallback) |
| Vectorisation | TF-IDF (sparse) and Word2Vec embeddings (dense) | scikit-learn, gensim |
| Topic modelling | LDA and NMF | scikit-learn |
| Reporting | top terms per topic, comparison, bar charts | matplotlib |

## Repository layout

```
municipal-complaints-nlp/
├── data/sample_complaints.csv     # synthetic sample so the project runs out of the box
├── src/
│   ├── config.py                  # paths + model parameters (edit here)
│   ├── make_sample_data.py        # regenerates the synthetic sample
│   ├── preprocess.py              # cleaning / tokenisation / lemmatisation
│   ├── vectorize.py               # TF-IDF + embeddings + comparison
│   ├── topic_models.py            # LDA + NMF + diversity metric
│   └── run_analysis.py            # end-to-end orchestration
├── outputs/                       # generated: topics.txt, *.png
├── requirements.txt
└── environment.yml
```

## How to use the code

1. **Create an environment and install dependencies**

   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   # or:  conda env create -f environment.yml && conda activate municipal-complaints-nlp
   ```

2. **Run the pipeline on the bundled sample**

   ```bash
   python src/run_analysis.py
   ```

   Results are printed to the console and written to `outputs/`:
   `topics.txt` (vectoriser comparison + LDA/NMF topics) and the
   `lda_topics.png` / `nmf_topics.png` charts.

3. **Use the real CFPB Consumer Complaint Database**

   a. Go to the CFPB data portal:
      <https://www.consumerfinance.gov/data-research/consumer-complaints/search/>
   b. Tick the filter **"Has narrative"** (only those rows contain free text),
      optionally narrow the date range to keep the file small, and download the
      **CSV**. (The full database is several GB; a filtered export or the Kaggle
      mirror is easier to handle.)
   c. Save it as `data/complaints.csv` in this project.
   d. In `src/config.py` set `DATASET = "cfpb"`. That switch already points the
      pipeline at `data/complaints.csv`, the `"Consumer complaint narrative"`
      column, and caps the load at `MAX_ROWS = 5000` (raise or set to `None`
      for all rows).
   e. Re-run `python src/run_analysis.py`.

   The config also removes CFPB's `XXXX` redactions and finance boilerplate as
   stop-words. After your first run, look at `outputs/topics.txt` and add any
   remaining noise words to `DOMAIN_STOPWORDS` in `config.py`, then re-run.

## Notes

- `data/sample_complaints.csv` is **synthetic** and exists only so the project
  runs without an external download. Replace it with real complaint data for a
  meaningful analysis.
- The code degrades gracefully: if NLTK data or gensim are unavailable it falls
  back to scikit-learn's stop-words and an LSA (Truncated SVD) embedding so the
  pipeline still runs anywhere.
