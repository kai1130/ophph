import math
import torch
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util


def confidence(cos_sim):
    percent_dist = (math.pi - math.acos(cos_sim)) * 100 / math.pi
    return percent_dist


model = SentenceTransformer('all-MiniLM-L6-v2')

st.header('Zero-Shot Text Classification')
st.caption('just a quick implementation :)')
st.markdown("""---""")

col1, col2 = st.columns(2)

with col1:
    message_input = st.text_area(
        label='Type Message',
        value="Where is VIN?",
        key="placeholder",
    )
    if message_input:
        message = message_input
        st.caption("Current Message")
        st.write(f'{message}')

with col2:
    category_input = st.text_area(
        label='Define Categories',
        value="What is the current location of VIN?\nWhat is the soft match date of VIN?\nWhat is the status of VIN?",
        key='categories',
    )

    if category_input:
        categories = [c.strip() for c in category_input.split('\n')]
        st.caption('Current Categories')
        st.write('  \n'.join(categories))

st.markdown("""---""")

columns = st.columns((1, 1, 1))
classify = columns[1].button('Classify Request')

if classify:

    message_embedding = model.encode([message])
    category_embeddings = model.encode(categories)

    categories_map = {k: v for k, v in enumerate(categories)}

    category_similarity = util.cos_sim(message_embedding, category_embeddings)

    categories_argmax = int(category_similarity.argmax(axis=1)[0])
    categories_str = categories_map.get(categories_argmax, None)

    categories_args = np.argsort(category_similarity.tolist()[0])
    
    categories_argstrs = [
        (categories_map.get(cat, None),
        category_similarity.tolist()[0][cat])
        for cat in categories_args[-5:-1]
    ]

    categories_max = float(category_similarity.max(axis=1)[0][0])
    categories_confidence = confidence(categories_max)

    st.metric(
        label="Predicted Category",
        value=categories_str,
        delta=f'Confidence: {categories_confidence:.2f}%'
    )
    for tup in categories_argstrs:
        st.text(tup)
st.markdown("""---""")
