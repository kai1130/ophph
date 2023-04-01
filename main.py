import math
import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util


def confidence(cos_sim):
    percent_dist = (math.pi - math.acos(cos_sim)) * 100 / math.pi
    return percent_dist


model = SentenceTransformer('all-MiniLM-L6-v2')

col1, col2 = st.columns(2)

with col1:
    message_input = st.text_input(
        label='Input message here',
        value='Hi, how much does your product cost?',
        key="placeholder"
    )
    if message_input:
        message = message_input
        st.write(f"Message: {message}")

with col2:
    category_input = st.text_input(
        label='Desired categories (separated by commas)',
        value='''
        Price or Product Inquiry, Order Placement, Appointment Scheduling
        ''',
        key='categories'
    )

    if category_input:
        categories = [c.strip() for c in category_input.split(',')]
        st.write('Categories')
        st.markdown(categories)

message_embedding = model.encode([message])
category_embeddings = model.encode(categories)

categories_map = {k: v for k, v in enumerate(categories)}

category_similarity = util.cos_sim(message_embedding, category_embeddings)

categories_argmax = int(category_similarity.argmax(axis=1)[0])
categories_str = categories_map.get(categories_argmax, None)

categories_max = float(category_similarity.max(axis=1)[0][0])
categories_confidence = confidence(categories_max)

st.write('AutoCategory: {categories_str} | Confidence: {categories_confidence}%')
