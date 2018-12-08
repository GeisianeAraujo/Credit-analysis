# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')

base.describe()

# verifica se tem números negativos
base.loc[base['age'] < 0]

# valores invalidos. Exemplo: coluna age com numero negativos
# apagar somente os registros com problema
#base.drop(base[base.age < 0].index, inplace=True)

# ou caso não tenha os dados, preencher os valores com a média de idade da base de dados.
base['age'][base.age > 0].mean()

base.loc[base.age < 0, 'age'] = 40.92

# valores inválidos "NULL"
#print para todos os registros
pd.isnull(base['age'])
#print dos dados inválidos
base.loc[pd.isnull(base['age'])]

# antes do tratamento, é necessária a divisão da base para aplicação futura do deep learn.   
# divisão em variaveis previsores e classe,separando as linhas.
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# tratar os valores nulos
from sklearn.preprocessing import Imputer
# valor a substituir NaN, estratégia média e axi = coluna
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
# aplicar o tratamento
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

# escalonamento de dados, quando o os valores estão em escalas distintas. Caso for utilizar o KNN (distância Euclidiana).
# importante: auxilia o algoritmo processar mais rápido.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)