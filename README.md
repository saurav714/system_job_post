# Smart Job Recommender Filter System

Demo project: a simple job recommendation app that filters out jobs you've applied or viewed and learns from clicks using semantic similarity.

Features
- Sample job dataset (CSV) with fields: job_id, title, company, skills, applied, viewed, clicked
- Uses `sentence-transformers` model `all-MiniLM-L6-v2` to compute semantic embeddings
- Feedback loop: clicking a job updates the user profile embedding, improving future recommendations
- Streamlit dashboard to view recommendations and mark jobs as Applied or Clicked
- Dynamic refresh after feedback

Requirements
- Python 3.8+
- Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the demo

```powershell
streamlit run app.py
```

Files
- `app.py` — Streamlit UI
- `recommender.py` — Recommendation logic, embedding persistence, feedback handling
- `data/jobs.csv` — demo dataset
- `requirements.txt` — python dependencies

Notes & tips
- The first run will download the `all-MiniLM-L6-v2` model (requires internet) and compute embeddings. This may take a minute.
- Embeddings are saved to `data/job_embeddings.npy` and a saved user profile is stored at `data/user_profile.npy`.
- To force a recompute of embeddings, delete `data/job_embeddings.npy` or touch the sidebar button (app provides a button to force recomputation via rerun).

Next steps (optional)
- Add user accounts and persistent per-user profiles
- Improve feedback weighting and decay
- Add more features (location, salary, remote flag) and richer filtering

License: MIT (demo educational project)
