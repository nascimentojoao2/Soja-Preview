from __future__ import annotations

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_price  # noqa: E402

st.set_page_config(page_title='Soybean Price Prediction', page_icon='🌱', layout='wide')

MODEL_PATH = BASE_DIR / 'models' / 'soybean_model.pkl'
PROCESSED_DATA_PATH = BASE_DIR / 'models' / 'processed_dataset.csv'
RAW_DATA_PATH = BASE_DIR / 'data' / 'soybean_prices.csv'
METRICS_PATH = BASE_DIR / 'models' / 'metrics.json'


def ensure_model_exists():
    if not MODEL_PATH.exists() or not PROCESSED_DATA_PATH.exists():
        st.error('Model not found. Run `python src/train_model.py` first.')
        st.stop()


@st.cache_data
def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


ensure_model_exists()
raw_df = load_raw_data()
processed_df = load_processed_data()
metrics = load_metrics()
artifact = load_artifact()

st.title('🌱 Soybean Price Prediction using Machine Learning')
st.caption('Projeto de portfólio em Machine Learning com dashboard em Streamlit.')

col1, col2, col3 = st.columns(3)
latest_price = raw_df.sort_values('date').iloc[-1]['price']
col1.metric('Último preço disponível', f'R$ {latest_price:.2f}')
col2.metric('Melhor modelo', artifact['best_model_name'])
col3.metric('Registros históricos', f"{len(raw_df):,}".replace(',', '.'))

st.sidebar.header('Prever preço futuro')
default_date = pd.to_datetime(raw_df['date'].max()) + pd.Timedelta(days=7)
future_date = st.sidebar.date_input('Escolha uma data', value=default_date)

if st.sidebar.button('Predict Price', use_container_width=True):
    pred = predict_price(pd.Timestamp(future_date))
    st.sidebar.success(f'Preço previsto: R$ {pred:.2f}')

left, right = st.columns((2, 1))
with left:
    st.subheader('Histórico de preços')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw_df['date'], raw_df['price'])
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.set_title('Preço da soja ao longo do tempo')
    st.pyplot(fig)

    st.subheader('Preço real vs previsto (conjunto de teste)')
    test_dates = processed_df.iloc[artifact['test_index']]['date'].reset_index(drop=True)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(test_dates, artifact['test_actuals'], label='Real')
    ax2.plot(test_dates, artifact['test_predictions'], label='Previsto')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Preço')
    ax2.set_title('Comparação entre preço real e previsto')
    ax2.legend()
    st.pyplot(fig2)

with right:
    st.subheader('Métricas dos modelos')
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader('Amostra do dataset')
    st.dataframe(raw_df.tail(10), use_container_width=True, hide_index=True)

    st.subheader('Observações')
    st.markdown(
        '- O dataset incluído é sintético, criado para fins de portfólio.\n'
        '- O modelo usa recursos de data, câmbio e variáveis auxiliares simuladas.\n'
        '- Para produção real, substitua pelos dados de mercado que você coletar.'
    )
