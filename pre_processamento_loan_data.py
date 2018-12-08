# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:12:46 2018

@author: GeisyPC
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

# Criando a base de dados de treinamento e a de teste
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# importação da biblioteca
# criação do classificador
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)