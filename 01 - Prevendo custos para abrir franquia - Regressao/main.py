import streamlit as st
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Previsao inicial de custo para franquia")

dados = pd.read_csv("slr12.csv", sep=";")

X = dados[['FrqAnual']]
y = dados['CusInic']


modelo = LinearRegression().fit(X,y)

col1, col2 = st.columns(2)

with col1:
  st.header("Dados")
  st.table(dados.head(10))


with col2:
  st.header(" Grafico de Dispersao")
  fig, ax = plt.subplots()
  ax.scatter(X, y, color="blue")
  ax.plot(X, modelo.predict(X), color="red")
  st.pyplot(fig)

st.header("Valor anual da franquia:")
novo_valor = st.number_input("Insira um novo valor", min_value=1.0, max_value=99999.0, value=1500.0, step=0.1)
processar = st.button("Processar")

if processar:
  dados_novo_valor = pd.DataFrame([[novo_valor]], columns=['FrqAnual'])
  prev = modelo.predict(dados_novo_valor)
  st.header(f"Previsao de custo inicial R$ {prev[0]:.2f}")