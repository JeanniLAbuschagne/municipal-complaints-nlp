"""
Preprocessing: turn raw complaint narratives into clean, lemmatised token strings.

Preferred stack: NLTK (WordNet lemmatiser + stop-word list).
If NLTK or its data are unavailable, the module falls back to scikit-learn's
built-in English stop-word list and skips lemmatisation, so the pipeline still
runs everywhere without an internet connection.
"""
import re
from config import DOMAIN_STOPWORDS

# --- Try to load the preferred NLTK components, else fall back --------------
try:
    import nltk
    from nltk.corpus import stopwords, wordnet  # noqa: F401
    from nltk.stem import WordNetLemmatizer

    # Make sure required corpora are present (downloads only if online).
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _BACKEND = "nltk"
except Exception:  # ImportError or missing data with no network
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOPWORDS = set(ENGLISH_STOP_WORDS)
    _LEMMATIZER = None
    _BACKEND = "sklearn-fallback"

STOPWORDS = _STOPWORDS | DOMAIN_STOPWORDS

_TOKEN_RE = re.compile(r"[a-z]+")
_URL_RE = re.compile(r"http\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Lower-case, strip URLs, and remove everything except alphabetic chars."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    """Tokenise, drop stop-words and very short tokens, then lemmatise."""
    tokens = _TOKEN_RE.findall(clean_text(text))
    out = []
    for tok in tokens:
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        if _LEMMATIZER is not None:
            tok = _LEMMATIZER.lemmatize(tok)
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def preprocess_corpus(texts):
    """Return (cleaned_strings, tokenised_docs) for a list of raw texts."""
    tokenised = [tokenize(t) for t in texts]
    cleaned = [" ".join(toks) for toks in tokenised]
    return cleaned, tokenised


def backend_name() -> str:
    return _BACKEND
