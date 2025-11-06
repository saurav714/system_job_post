import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import recommender
import pytest

# Dummy model to avoid downloading real sentence-transformers during tests
class DummyModel:
    def __init__(self, dim=8):
        self._dim = dim

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        # accept both list[str] and list[list]
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            h = sum(ord(c) for c in str(t))
            vec = np.array([((h >> i) & 255) / 255.0 for i in range(self._dim)])
            out.append(vec)
        return np.array(out)

    def get_sentence_embedding_dimension(self):
        return self._dim


def setup_temp_data():
    # create a temp dir and copy sample csv there
    tmp = tempfile.mkdtemp()
    src = os.path.join(os.path.dirname(__file__), '..', 'data', 'jobs.csv')
    dst_dir = tmp
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'jobs.csv')
    shutil.copyfile(src, dst)
    return tmp


def test_filtering_applied_viewed():
    tmp = setup_temp_data()
    # point recommender to temp data dir
    recommender.DATA_DIR = tmp
    recommender.JOBS_CSV = os.path.join(tmp, 'jobs.csv')
    recommender.EMBED_PATH = os.path.join(tmp, 'job_embeddings.npy')
    recommender.USER_PROFILE_PATH = os.path.join(tmp, 'user_profile.npy')

    # use dummy model
    recommender._model = None
    recommender.get_model = lambda: DummyModel()

    # compute embeddings
    embeddings, df = recommender.embed_jobs(force_recompute=True)

    # mark first job applied and second viewed
    first_id = df.iloc[0]['job_id']
    second_id = df.iloc[1]['job_id']
    recommender.mark_applied(first_id)
    recommender.mark_viewed(second_id)

    recs = recommender.get_recommendations(top_k=20)
    ids = set(recs['job_id'].tolist())
    assert first_id not in ids, 'Applied job should be filtered out'
    assert second_id not in ids, 'Viewed job should be filtered out by default'

    shutil.rmtree(tmp)


def test_recommendations_non_empty_for_new_users():
    tmp = setup_temp_data()
    recommender.DATA_DIR = tmp
    recommender.JOBS_CSV = os.path.join(tmp, 'jobs.csv')
    recommender.EMBED_PATH = os.path.join(tmp, 'job_embeddings.npy')
    recommender.USER_PROFILE_PATH = os.path.join(tmp, 'user_profile.npy')

    recommender._model = None
    recommender.get_model = lambda: DummyModel()

    embeddings, df = recommender.embed_jobs(force_recompute=True)

    recs = recommender.get_recommendations(top_k=5)
    assert len(recs) > 0, 'Recommendations should return results for new users'

    shutil.rmtree(tmp)


def test_feedback_loop_changes_ranking():
    tmp = setup_temp_data()
    recommender.DATA_DIR = tmp
    recommender.JOBS_CSV = os.path.join(tmp, 'jobs.csv')
    recommender.EMBED_PATH = os.path.join(tmp, 'job_embeddings.npy')
    recommender.USER_PROFILE_PATH = os.path.join(tmp, 'user_profile.npy')

    recommender._model = None
    recommender.get_model = lambda: DummyModel()

    embeddings, df = recommender.embed_jobs(force_recompute=True)

    # use a textual query to create a non-zero profile
    query = 'python'
    before = recommender.get_recommendations(query_text=query, top_k=10)
    before_scores = before['score'].tolist()

    # choose a job to click (pick the 3rd recommended)
    if len(before) < 3:
        pytest.skip('Not enough jobs for the test')
    clicked_job = before.iloc[2]['job_id']

    recommender.register_click(clicked_job, session_state={})

    after = recommender.get_recommendations(query_text=query, top_k=10)
    after_scores = after['score'].tolist()

    # Scores should change after feedback (ranking should adapt). At least one score differs.
    assert before_scores != after_scores, 'Scores/ranking should change after clicking'

    shutil.rmtree(tmp)
