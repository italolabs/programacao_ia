# Importar módulos
import streamlit as st #lib que transforma python em sites web
import joblib #salva e exporta o modelo treinado de IA em um binário
import numpy as np #lib para organizar dados numéricos


#Passo1: configurando a aba do navegador
st.set_page_config(page_title="Análise de Churn",page_icon="🤖")

#textos da tela principal
st.title("Sistema de retenção de base ") #titulo da página
st.markdown("Insira dados do cliente para verificar risco de cancelamento") #subtitulo

#passo 2: importar os dados da inteligência artificial com o joblib
modelo = joblib.load("modelo_churn_v1.pkl") #carrega as regras de decisão do modelo
scaler = joblib.load("padronizador_v1.pkl") #carrega a régua matemática

#passo 3: criar a interface de entrada com um formulário
col1, col2 = st.columns(2) #criando duas colunas

#coluna lado esquerdo (col1)
with col1:
    tempo = st.number_input("Tempo de contrato (meses)", min_value=1, value=12)
    valor = st.number_input("Valor da assinatura: (R$)", min_value=.0, value=50.0)

with col2:
        reclamacoes = st.slider("Histórico de reclamações", 0,10,1)
    
#passo 4: processamento de dados
if st.button("🔍 Analisar Risco"):
    dados = scaler.transform([[tempo, valor, reclamacoes]])
    probabilidade = modelo.predict_proba(dados)[0][1] #previsão de probabilidade

#passo 5: feedback de negócios
    st.divider() #cria uma linha divisória

#probabilidade <70%
    if probabilidade >0.7:
        st.error(f"*ALTO RISCO DE CHURN* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação:* Oferecer cupom de fidelidade FID210360OFF")
    
    elif probabilidade >0.3:
        st.warning(f"*Risco moderado de churn* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação: *Realizar chamada de companhamento.")
    else:
        st.success(f"*Cliente estável* ({probabilidade*100:.1f}%)")
        st.info("Nada a realizar no momento.")
        