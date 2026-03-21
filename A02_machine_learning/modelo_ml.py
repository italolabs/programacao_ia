""" ETAPA 01 """

# Importando todos os módulos necessários
import pandas as pd # ferramenta para criar e alterar dados em tabelas
import numpy as np # ferramenta de análise matemática

from sklearn.preprocessing import StandardScaler # responsável por organizar os números para que fiquem todos na mesma escala
from sklearn.ensemble import RandomForestClassifier # 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as pyplot
import joblib

""" ETAPA 02 """

# Importação do dataset

try:
    print()
    print("Carregando arquivo 'churn_data.csv'...")
    print()
    df = pd.read_csv("churn_data.csv") # ler o arquivo e criar uma tabela
    print()
    print(f"Sucesso, {len(df)} linhas importadas.")
    
except FileNotFoundError:
    print("O arquivo não pode ser encontrado na pasta.")
    exit()
    
""" ETAPA 03 """

# Pré processamento de dados (preparar a ia para ser treinada)

# Passo 1: Separar pergunta (x) da resposta (y)

# (x) -> tudo menos a coluna cancelou, são as "pistas" pro modelo
X = df.drop("cancelou",axis=1)

# (y) -> apenas a coluna "cancelou", é o que queremos que o modelo preveja
y = df["cancelou"]

# Passo 2: Dividir o treino do teste 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42) # test_size=0.2 separa 20% da massa de dados para testar o modelo

# Passo 3: Normalizando (colocando tudo na mesma escala)
scaler = StandardScaler()

# fit transform do treino: IA calcula média e desvio padrão do treino
X_train_scaled = scaler.fit_transform(X_train)

# fit transform no teste: usamos a régua calculada no treino
X_test_scaled=scaler.transform(X_test)

""" ETAPA 04 """

# Treinar o modelo e realizar a previsão de dados

# Criando modelo
modelo_churn = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators=100 cria 100 árvores de decisão

# Treinas/Ajustar a IA
modelo_churn.fit(X_train_scaled, y_train)

# Prever as respostas
previsoes = modelo_churn.predict(X_test_scaled)

""" ETAPA 05 """

# Avaliação do modelo
print()
print("                   Relatório de Performance")
print()
print(classification_report(y_test,previsoes))

""" ETAPA 06 """

# Deploy -> salvar o trabalho
joblib.dump(modelo_churn,"modelo_churn_v1.pkl")

joblib.dump(scaler, "padronizador_v1.pkl")
print("Arquivo de ML foram exportados com sucesso")