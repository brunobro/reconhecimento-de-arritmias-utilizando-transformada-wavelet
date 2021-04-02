# -*- coding: utf-8 -*-
'''
3. Validação: Emprega as máquinas induzidas no script 2_inducao.py para validar
os modelos sobre os dados de teste
'''
from sklearn.metrics import precision_score, accuracy_score, recall_score
import numpy as np
import joblib
import pickle
import config

'''
Lê os atributos salvos e os rótulos respectivos
'''
atributos_0 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_validacao_0.pkl', 'rb'))
atributos_1 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_validacao_1.pkl', 'rb'))
atributos_2 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_validacao_2.pkl', 'rb'))
rotulos     = pickle.load(open(config.DIR_ATRIBUTOS + 'rotulos_validacao.pkl', 'rb'))

'''
Lê os pesos atribuídos a cada máquina induzida
'''
pesos = pickle.load(open(config.DIR_MODELOS + 'pesos.pkl', 'rb'))

'''
Carrega as máquinas (modelos) de reconhecimento de padrões cardíacos
'''
clf_0 = joblib.load(config.DIR_MODELOS + 'clf_0.jl')
clf_1 = joblib.load(config.DIR_MODELOS + 'clf_1.jl')
clf_2 = joblib.load(config.DIR_MODELOS + 'clf_2.jl')

'''
Faz as predições utilizando a Máquina de Aprendizado induzidas
'''
predicoes_0 = clf_0.predict(atributos_0)
predicoes_1 = clf_1.predict(atributos_1)
predicoes_2 = clf_2.predict(atributos_1)

predicoes_0_proba = clf_0.predict_proba(atributos_0)
predicoes_1_proba = clf_1.predict_proba(atributos_1)
predicoes_2_proba = clf_2.predict_proba(atributos_2)

'''
Implementa um comitê de voto suave
Calcula a soma das probabilidades de predição, ponderada pelos pesos 
obtidos em 2_inducao.py
'''
soma_probabilidade_classe_negativa = pesos[0] * predicoes_0_proba[:,0] + pesos[1] * predicoes_1_proba[:,0] + pesos[2] * predicoes_2_proba[:,0]

soma_probabilidade_classe_positiva = pesos[0] * predicoes_0_proba[:,1] + pesos[1] * predicoes_1_proba[:,1] + pesos[2] * predicoes_2_proba[:,1]

#Para armazenar as predições do comitê de máquinas
predicoes = np.zeros(soma_probabilidade_classe_negativa.shape[0])

for i in range(0, soma_probabilidade_classe_negativa.shape[0]):
	
    #Classe predita pelo comitê
    if soma_probabilidade_classe_negativa[i] < soma_probabilidade_classe_positiva[i]:
        predicoes[i] = 1

print('\nResultados sobre o conjunto de teste')

print('M0')
print(' > Acurácia', accuracy_score(rotulos, predicoes_0))
print(' > Precisão', precision_score(rotulos, predicoes_0))
print(' > Recobrimento', recall_score(rotulos, predicoes_0))

print('M1')
print(' > Acurácia', accuracy_score(rotulos, predicoes_1))
print(' > Precisão', precision_score(rotulos, predicoes_1))
print(' > Recobrimento', recall_score(rotulos, predicoes_1))

print('M2')
print(' > Acurácia', accuracy_score(rotulos, predicoes_2))
print(' > Precisão', precision_score(rotulos, predicoes_2))
print(' > Recobrimento', recall_score(rotulos, predicoes_2))

print('Comitê de Máquinas')
print(' > Acurácia', accuracy_score(rotulos, predicoes))
print(' > Precisão', precision_score(rotulos, predicoes))
print(' > Recobrimento', recall_score(rotulos, predicoes))
