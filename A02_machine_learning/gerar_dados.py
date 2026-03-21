
# Importando as libs necessárias

import pandas as pd
import numpy as np

# Criando números aleatórios para simular dados reais
# Definindo uma semente parafins de simulação

np.random.seed(42)

# Gerando 500 registros
n_registros = 500

# Estruturando os dados do arquivo .csv
data = {
    "tempo_contrato": np.random.randint(1, 48, n_registros), # 1 a 48 meses
    "valor_mensal": np.random.uniform(50.0, 150.0, n_registros).round(2), # Assinatura com valores que variam de 50 a 150 dinheiros
    "reclamacoes": np.random.poisson(1.5, n_registros) # Cada usuário tem uma média de 1.5 reclamações
}

# Convertendo a estrutura de dicionário em um conjunto de dados
df = pd.DataFrame(data)

# Criar a simulação da lógica de churn
# O cliente tem mais chance de sair se tiver muitas reclamações OU se o contrato for curto
df["cancelou"]=((df["reclamacoes"]>2)|(df["tempo_contrato"]<6)).astype(int)

# Salvando o dataset em .csv
df.to_csv("churn_data.csv", index=False)
print("Arquivo 'churn_data.csv' gerado com sucesso!")
