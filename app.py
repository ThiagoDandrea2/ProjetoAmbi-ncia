import streamlit as st
import matplotlib.pyplot as plt
from analysis import (
    carregar_dados, 
    plot_evolucao, 
    plot_correlacao, 
    prever_peso,
    carregar_modelo
)

# Configuração da página
st.set_page_config(page_title="Dashboard Ambiência Animal", layout="wide")

# Carregar dados e modelo
df = carregar_dados()
model = carregar_modelo()

# Sidebar - Filtros
st.sidebar.header("Filtros")
dias = st.sidebar.slider("Número de dias", 7, 30, 15)
variavel = st.sidebar.selectbox("Variável para análise", df.columns[1:])

# Página principal
st.title("Dashboard de Ambiência Animal")

# Gráficos
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Evolução do {variavel.replace('_', ' ').title()}")
    fig = plot_evolucao(df, variavel, dias)
    st.pyplot(fig)

with col2:
    st.subheader("Correlação entre Variáveis")
    corr_col = st.selectbox("Selecione para correlacionar", df.columns[1:-1], key='corr')
    fig = plot_correlacao(df, variavel, corr_col)
    st.pyplot(fig)

# Predição
st.subheader("Predição de Peso")
col1, col2, col3, col4 = st.columns(4)
temp = col1.number_input("Temperatura (°C)", value=25.0)
umid = col2.number_input("Umidade (%)", value=60.0)
agua = col3.number_input("Água Consumida (L)", value=15.0)
racao = col4.number_input("Ração Consumida (kg)", value=8.0)

if st.button("Prever Peso"):
    pred = prever_peso(model, temp, umid, agua, racao)
    st.success(f"Peso previsto: {pred:.2f} kg")

# Dados brutos
if st.checkbox("Mostrar dados brutos"):
    st.dataframe(df.tail(dias))