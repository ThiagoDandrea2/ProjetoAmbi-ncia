import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

def criar_dataset_exemplo():
    """Cria um dataset de exemplo caso não exista"""
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    
    # Criar dados com algumas relações conhecidas para a regressão
    temperatura = np.random.normal(25, 3, 30)
    umidade = np.random.normal(60, 10, 30)
    agua_consumida = np.random.normal(15, 2, 30) + 0.5*temperatura
    racao_consumida = np.random.normal(8, 1, 30)
    
    # Peso com dependência das outras variáveis
    peso = 50 + 1.5*racao_consumida + 0.3*agua_consumida - 0.2*temperatura + np.random.normal(0, 2, 30)
    peso = peso.cumsum()  # Efeito cumulativo
    
    data = {
        'data': [d.date() for d in dates],
        'temperatura': temperatura,
        'umidade': umidade,
        'agua_consumida': agua_consumida,
        'racao_consumida': racao_consumida,
        'peso': peso
    }
    
    return pd.DataFrame(data)

def carregar_dados():
    """Carrega os dados do arquivo CSV ou cria um novo dataset"""
    if not os.path.exists('data/dados_sensor.csv'):
        os.makedirs('data', exist_ok=True)
        df = criar_dataset_exemplo()
        df.to_csv('data/dados_sensor.csv', index=False)
    return pd.read_csv('data/dados_sensor.csv')

def treinar_modelo(df):
    """
    Treina um modelo de regressão linear para predição de peso
    Retorna o modelo treinado e as métricas de avaliação
    """
    # Preparar dados
    X = df[['temperatura', 'umidade', 'agua_consumida', 'racao_consumida']]
    y = df['peso']
    
    # Dividir em treino e teste (usando os últimos 5 dias para teste)
    X_train, X_test = X.iloc[:-5], X.iloc[-5:]
    y_train, y_test = y.iloc[:-5], y.iloc[-5:]
    
    # Treinar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {'MAE': mae, 'R2': r2}

def salvar_modelo(model, path='models/modelo_peso.pkl'):
    """Salva o modelo treinado no caminho especificado"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def carregar_modelo(path='models/modelo_peso.pkl'):
    """Carrega um modelo salvo"""
    return joblib.load(path)

def plot_evolucao(df, variavel, dias=15):
    """Gera um gráfico de linha da evolução de uma variável"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['data'].tail(dias), df[variavel].tail(dias), marker='o')
    ax.set_title(f"Evolução do {variavel.replace('_', ' ').title()}")
    ax.set_xlabel('Data')
    ax.set_ylabel(variavel.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_correlacao(df, var1, var2):
    """Gera um gráfico de dispersão mostrando a correlação entre duas variáveis"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[var1], df[var2], alpha=0.6)
    ax.set_title(f"Correlação entre {var1.replace('_', ' ')} e {var2.replace('_', ' ')}")
    ax.set_xlabel(var1.replace('_', ' '))
    ax.set_ylabel(var2.replace('_', ' '))
    
    # Calcular e mostrar coeficiente de correlação
    corr = df[[var1, var2]].corr().iloc[0,1]
    ax.annotate(f'Correlação: {corr:.2f}', xy=(0.7, 0.9), xycoords='axes fraction')
    
    plt.tight_layout()
    return fig

def prever_peso(model, temperatura, umidade, agua, racao):
    """Faz uma predição de peso com base nas variáveis de entrada"""
    dados_entrada = [[temperatura, umidade, agua, racao]]
    return model.predict(dados_entrada)[0]