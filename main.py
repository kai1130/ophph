import math
import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

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

