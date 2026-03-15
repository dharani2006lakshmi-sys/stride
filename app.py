"""
STRIDE — Structural Time-based Risk Intelligence for Damage Evolution
Dashboard: Zone-centric structural health monitoring.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from ingestion.simulated_loader import load_structural_data
from ingestion.iot_loader import load_iot_data
from features.processor import compute_features
from model.risk_model import RiskModel
from decision.logic import apply_decision_logic
from database.db_manager import init_db

# initialize database
init_db()

st.set_page_config(
    page_title="STRIDE Structural Risk Dashboard",
    layout="wide"
)

st.title("🏗 STRIDE — Structural Health Monitoring")

data_source = st.sidebar.selectbox(
    "DATA SOURCE",
    ["Simulated Data", "IoT Data"]
)

if data_source == "Simulated Data":
    df = load_structural_data()
else:
    df = load_iot_data()

st.subheader("Raw Sensor Data")
st.dataframe(df)

features = compute_features(df)

model = RiskModel()

risk = model.predict(features)

decision = apply_decision_logic(risk)

st.subheader("Risk Result")

st.metric("Structural Risk Index", round(risk,3))

st.write("Decision:", decision)

st.line_chart(df)
