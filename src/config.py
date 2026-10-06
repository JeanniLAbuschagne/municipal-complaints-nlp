"""Central configuration for the municipal-complaints NLP pipeline.

To switch between the bundled synthetic sample and the real CFPB dataset,
change DATASET below. Everything else adapts automatically.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# ---- Dataset selection -----------------------------------------------------
# "sample" -> bundled synthetic data (runs offline, no download)
# "cfpb"   -> real CFPB Consumer Complaint Database (see README for download)
DATASET = "cfpb"

if DATASET == "cfpb":
    # Put the downloaded CFPB CSV here (any name works; update if different).
    DATA_PATH = os.path.join(ROOT, "data", "complaints.csv")
    TEXT_COLUMN = "Consumer complaint narrative"
    # CFPB is huge; cap how many rows with a narrative we load (None = all).
    MAX_ROWS = 5000
else:  # "sample"
    DATA_PATH = os.path.join(ROOT, "data", "sample_complaints.csv")
    TEXT_COLUMN = "narrative"
    MAX_ROWS = None

# ---- Modelling parameters --------------------------------------------------
N_TOPICS = 8           # number of latent topics to extract
N_TOP_TERMS = 10       # terms shown per topic
MIN_DF = 5             # ignore terms appearing in < MIN_DF documents
MAX_DF = 0.5           # ignore terms appearing in > 50% of documents
NGRAM_RANGE = (1, 2)   # unigrams + bigrams
RANDOM_STATE = 42

# Extra domain stop-words that are too generic to be useful as topics.
DOMAIN_STOPWORDS = {
    # generic complaint language (applies to both datasets)
    "complaint", "complain", "writing", "report", "council", "municipality",
    "please", "request", "wish", "formally", "ratepayer", "repeatedly",
    "third", "time", "despite", "several", "calls", "unacceptable", "frustrated",
    # CFPB-specific noise: "XXXX" redactions and finance boilerplate
    "xxxx", "xx", "xxxxxxxx", "account", "company", "credit", "would", "received",
    "told", "said", "also", "get", "got", "us",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
