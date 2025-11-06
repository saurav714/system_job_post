import streamlit as st
import pandas as pd
from recommender import get_recommendations, register_click, mark_applied, mark_viewed, load_jobs, embed_jobs

st.set_page_config(page_title='Smart Job Recommender Filter System', layout='wide')

st.title('Smart Job Recommender Filter System')

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = None

with st.sidebar:
    st.header('Controls')
    query = st.text_input('Enter your role/interest (optional)', value='')
    top_k = st.slider('Number of recommendations', 1, 20, 8)
    include_viewed = st.checkbox('Include previously viewed', value=False)
    recompute_embeddings = st.button('Recompute Embeddings (force)')

# Show some dataset stats
jobs_df = load_jobs()
col1, col2, col3 = st.columns(3)
col1.metric('Total jobs', int(len(jobs_df)), '')
col2.metric('Applied', int(jobs_df['applied'].sum()), '')
col3.metric('Clicked', int(jobs_df['clicked'].sum()), '')

# If user asked to recompute embeddings, force it and rerun
if recompute_embeddings:
    with st.spinner('Recomputing embeddings (this may take a while)...'):
        embed_jobs(force_recompute=True)
    st.experimental_rerun()

# Get recommendations
with st.spinner('Computing recommendations...'):
    recs = get_recommendations(session_state=st.session_state, query_text=query if query else None, top_k=top_k, include_viewed=include_viewed)

st.subheader('Recommended Jobs')

# (recompute handled above)

if recs.empty:
    st.write('No recommendations available. Try changing the query or unchecking filters.')
else:
    for _, row in recs.iterrows():
        job_id = row['job_id']
        # mark as viewed once it's presented so it won't be recommended again by default
        try:
            mark_viewed(job_id)
        except Exception:
            pass
        title = row['title']
        company = row['company']
        skills = row.get('skills', '')
        score = row.get('score', 0.0)

        card = st.container()
        with card:
            cols = st.columns([8,1,1])
            with cols[0]:
                st.markdown(f"### {title}  ")
                st.markdown(f"**{company}**  ")
                st.markdown(f"**Skills:** {skills}")
                st.write(f"Score: {score:.4f}")
            with cols[1]:
                applied_key = f"applied_{job_id}"
                if st.button('Applied', key=applied_key):
                    mark_applied(job_id)
                    st.experimental_rerun()
            with cols[2]:
                clicked_key = f"clicked_{job_id}"
                if st.button('Clicked', key=clicked_key):
                    # register click updates profile and saved CSV
                    register_click(job_id, session_state=st.session_state)
                    # mark viewed as well
                    mark_viewed(job_id)
                    st.experimental_rerun()

st.markdown('---')
st.subheader('All Jobs (debug)')
st.dataframe(load_jobs())

st.markdown('---')
st.caption('Demo app: recommendations will improve when you click jobs. Clicked jobs update the user profile embedding.')
