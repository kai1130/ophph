import math
import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util


def confidence(cos_sim):
    percent_dist = (math.pi - math.acos(cos_sim)) * 100 / math.pi
    return percent_dist


model = SentenceTransformer('all-MiniLM-L6-v2')

st.header('Zero-Shot Text Classification')
st.caption('just a quick out-of-the-box implementation :)')
st.markdown("""---""")

col1, col2 = st.columns(2)

with col1:
    st.subheader('Type a Message')
    message_input = st.text_area(
        label='Input message here',
        value='Hi, how much does your product cost?',
        key="placeholder"
    )
    if message_input:
        message = message_input
        st.caption(f"Current Message")
        st.write(f'{message}')

with col2:
    st.subheader('Define Categories')
    category_input = st.text_area(
        label='Desired categories (separated by commas)',
        value="Price or Product Inquiry\nOrder Placement\nAppointment Scheduling",
        key='categories'
    )

    if category_input:
        categories = [c.strip() for c in category_input.split('\n')]
        st.caption('Current Categories')
        st.write(category_input)

st.markdown("""---""")

columns = st.columns((1, 1, 1))
classify = columns[1].button('AutoClassify Message!')

if classify:

    message_embedding = model.encode([message])
    category_embeddings = model.encode(categories)

    categories_map = {k: v for k, v in enumerate(categories)}

    category_similarity = util.cos_sim(message_embedding, category_embeddings)

    categories_argmax = int(category_similarity.argmax(axis=1)[0])
    categories_str = categories_map.get(categories_argmax, None)

    categories_max = float(category_similarity.max(axis=1)[0][0])
    categories_confidence = confidence(categories_max)

    st.metric(
        label="Predicted Category",
        value=categories_str,
        delta=f'Confidence: {categories_confidence:.2f}%'
    )
st.markdown("""---""")
