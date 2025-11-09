import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from user_profiles import UserProfile, get_user_profile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JOBS_CSV = os.path.join(DATA_DIR, "jobs.csv")
EMBED_PATH = os.path.join(DATA_DIR, "job_embeddings.npy")
USER_PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.npy")

MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-load model
_model = None

# Logger
logger = logging.getLogger("smart_job_recommender")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load model {MODEL_NAME}: {e}")
            raise
    return _model


def load_jobs():
    """Load jobs from CSV with validation of required columns."""
    required_columns = ['job_id', 'title', 'company', 'skills']
    optional_columns = ['location', 'seniority', 'salary_range', 'applied', 'viewed', 'clicked']
    
    try:
        df = pd.read_csv(JOBS_CSV, dtype={"job_id": str}, encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading jobs CSV: {e}")
        # Create empty DataFrame with required columns
        df = pd.DataFrame(columns=required_columns + optional_columns)
    
    # Validate required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in jobs.csv: {missing_cols}")
    
    # Ensure boolean columns
    for col in ["applied", "viewed", "clicked"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    return df


def embed_jobs(force_recompute=False):
    """Compute or load job embeddings.
    Returns (embeddings ndarray, jobs_df)
    """
    df = load_jobs()
    # Clean column names from BOMs/whitespace and normalize
    df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
    # Ensure essential text columns exist
    for col in ['title', 'company', 'skills']:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing from jobs CSV — filling with empty strings")
            df[col] = ''
    model = get_model()
    if os.path.exists(EMBED_PATH) and not force_recompute:
        try:
            embeddings = np.load(EMBED_PATH)
            if embeddings.shape[0] == len(df):
                return embeddings, df
        except Exception:
            pass
    # Compose text to embed
    texts = (df['title'].fillna('') + ' - ' + df['company'].fillna('') + ' - ' + df['skills'].fillna('')).tolist()
    try:
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    except TypeError:
        # fallback if the model has a different signature
        embeddings = model.encode(texts)
        embeddings = np.array(embeddings)
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        raise
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(EMBED_PATH, embeddings)
    return embeddings, df


def get_user_profile(session_state, query_text=None):
    """Return the user profile embedding. Priority:
       1) session_state['user_profile'] if present
       2) saved user profile file
       3) embedding of query_text
       4) mean of clicked jobs
    """
    model = get_model()

    # 1. session state
    if session_state is not None and 'user_profile' in session_state and session_state['user_profile'] is not None:
        return session_state['user_profile']

    # 2. saved profile
    if os.path.exists(USER_PROFILE_PATH):
        try:
            return np.load(USER_PROFILE_PATH)
        except Exception:
            pass

    # 3. query text
    if query_text:
        return model.encode([query_text])[0]

    # 4. fallback: mean of job embeddings of clicked jobs
    embeddings, df = embed_jobs()
    clicked = df[df['clicked'] == True]
    if len(clicked) > 0:
        idxs = clicked.index.tolist()
        return embeddings[idxs].mean(axis=0)

    # 5. final fallback: zero vector
    dim = model.get_sentence_embedding_dimension()
    return np.zeros(dim)


def save_user_profile(emb):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(USER_PROFILE_PATH, emb)


def register_click(job_id, session_state=None):
    """Mark job as clicked and update user profile by adding the job embedding."""
    embeddings, df = embed_jobs()
    job_idx = df.index[df['job_id'] == str(job_id)].tolist()
    if not job_idx:
        return False
    idx = job_idx[0]
    df.at[idx, 'clicked'] = True
    df.to_csv(JOBS_CSV, index=False)

    job_emb = embeddings[idx]
    # update profile: weighted average (simple additive)
    profile = get_user_profile(session_state=session_state)
    # if profile is zero vector, replace; else average
    try:
        if np.linalg.norm(profile) == 0:
            new_profile = job_emb
        else:
            new_profile = (profile + job_emb) / 2.0
    except Exception:
        new_profile = job_emb
    if session_state is not None:
        session_state['user_profile'] = new_profile
    save_user_profile(new_profile)
    return True


def mark_applied(job_id):
    df = load_jobs()
    job_idx = df.index[df['job_id'] == str(job_id)].tolist()
    if not job_idx:
        return False
    idx = job_idx[0]
    df.at[idx, 'applied'] = True
    df.to_csv(JOBS_CSV, index=False)
    return True


def get_recommendations(session_state=None, query_text=None, top_k=10, include_viewed=False):
    embeddings, df = embed_jobs()
    user_profile = get_user_profile(session_state=session_state, query_text=query_text)
    # ensure user_profile is a valid array (get_user_profile may return None)
    if user_profile is None:
        try:
            model = get_model()
            dim = model.get_sentence_embedding_dimension()
            user_profile = np.zeros(dim)
        except Exception:
            # fallback dimension
            user_profile = np.zeros(384)

    # if user_profile is zero vector and query_text exists, prefer query
    if np.linalg.norm(user_profile) == 0 and query_text:
        model = get_model()
        user_profile = model.encode([query_text])[0]

    sims = cosine_similarity([user_profile], embeddings)[0]
    df = df.copy()
    df['score'] = sims

    # filter out applied (and optionally viewed)
    df = df[~df['applied']]
    if not include_viewed:
        df = df[~df['viewed']]

    df = df.sort_values('score', ascending=False)
    return df.head(top_k)


def mark_viewed(job_id):
    df = load_jobs()
    job_idx = df.index[df['job_id'] == str(job_id)].tolist()
    if not job_idx:
        return False
    idx = job_idx[0]
    df.at[idx, 'viewed'] = True
    df.to_csv(JOBS_CSV, index=False)
    return True


if __name__ == '__main__':
    def run_tests():
        """Run pytest programmatically against the tests/test_recommender.py file."""
        try:
            import pytest
        except Exception:
            logger.error('pytest is not installed. Install with: pip install pytest')
            raise
        rv = pytest.main([os.path.join(os.path.dirname(__file__), 'tests', 'test_recommender.py')])
        if rv == 0:
            print('✅ All tests passed')
        else:
            print('❌ Some tests failed')
        return rv

    # Allow running tests with `python recommender.py test`
    if len(sys.argv) > 1 and sys.argv[1] in ('test', 'tests'):
        run_tests()
    else:
        # quick smoke test
        logger.info('Loading embeddings (this will download model if needed)...')
        try:
            emb, df = embed_jobs()
            logger.info(f'Jobs: {len(df)}')
            logger.info(f'Embedding shape: {emb.shape}')
        except Exception as e:
            logger.error(f'Smoke test failed: {e}')
