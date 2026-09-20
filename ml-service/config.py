"""
PriceIQ ML Service — Central Configuration
All hyperparameters and paths in one place.
"""
import os
import sys

# Ensure UTF-8 stdout/stderr on Windows to avoid cp1252 charmap encoding errors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
MAPPINGS_DIR = os.path.join(ARTIFACTS_DIR, "mappings")

# ── Data Pipeline ────────────────────────────────────────────────────────────
MIN_SESSION_LEN = 2          # minimum interactions to form a usable session
MAX_SEQ_LEN = 20             # truncate sequences longer than this
PRODUCT_INTERACTION_TYPES = ["page_view", "add_to_cart", "wishlist_add", "purchase"]

# ── Temporal Split Ratios ────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Model Architecture ──────────────────────────────────────────────────────
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_GRU_LAYERS = 1
DROPOUT = 0.2

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 5                 # early-stopping patience (epochs)
RANDOM_SEED = 42

# ── Vocabulary Special Tokens ────────────────────────────────────────────────
PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS = 2           # PAD + UNK

# ── Synthetic Data ───────────────────────────────────────────────────────────
SYNTHETIC_NUM_USERS = 500
SYNTHETIC_NUM_SESSIONS = 2000
SYNTHETIC_EVENTS_PER_SESSION_RANGE = (3, 15)
SYNTHETIC_DAYS_SPAN = 30

# ── Inference ────────────────────────────────────────────────────────────────
COLD_START_THRESHOLD = 3     # min interactions before using GRU
TFIDF_THRESHOLD = 1          # min interactions for TF-IDF (1-2 range uses TF-IDF)

# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/priceiq")
DB_NAME = "priceiq"
