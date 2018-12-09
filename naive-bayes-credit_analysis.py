# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:31:45 2018

@author: GeisyPC
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
# constroi a tabela de probabilidades
classificador.fit(previsores_treinamento, classe_treinamento)
# previsões dos registros
previsoes = classificador.predict(previsores_teste)

# Visualização do percentual/quantidade de erros do registos de previsãos 
from sklearn.metrics import confusion_matrix, accuracy_score
# mostra o percentual de acerto
precisao = accuracy_score(classe_teste, previsoes)
# quantidade de acerto e erro
matriz = confusion_matrix(classe_teste, previsoes)