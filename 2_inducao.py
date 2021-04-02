# -*- coding: utf-8 -*-
'''
2. Indução: Induz três máquina de aprendizado utilizando os atributos 
gerados no script 1_extracao_atributos.py e o algoritmo Naive Bayes
'''
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
import numpy as np
import joblib
import pickle
import config

'''
Lê os atributos salvos
'''
atributos_0 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_inducao_0.pkl', 'rb'))
atributos_1 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_inducao_1.pkl', 'rb'))
atributos_2 = pickle.load(open(config.DIR_ATRIBUTOS + 'atributos_inducao_2.pkl', 'rb'))
rotulos     = pickle.load(open(config.DIR_ATRIBUTOS + 'rotulos_inducao.pkl', 'rb'))

'''
Treinamento da Máquina de Aprendizado e predições sobre o mesmo conjunto
'''
clf_0 = GaussianNB().fit(atributos_0, rotulos)
predicoes_0 = clf_0.predict(atributos_0)

clf_1 = GaussianNB().fit(atributos_1, rotulos)
predicoes_1 = clf_1.predict(atributos_1)

clf_2 = GaussianNB().fit(atributos_2, rotulos)
predicoes_2 = clf_2.predict(atributos_2)

'''
Com base na quantidade de falsos positivos (FP) e falsos negativos (FN) atribui pesos as máquinas de inferências
'''
print('Resultados Indivuais')

print('M0')
acc0 = accuracy_score(rotulos, predicoes_0)
print(' > Acurácia', acc0)
print(' > Precisão', precision_score(rotulos, predicoes_0))
print(' > Recobrimento', recall_score(rotulos, predicoes_0))
print(' > Matriz de confusão:', confusion_matrix(rotulos, predicoes_0))

print('M1')
acc1 = accuracy_score(rotulos, predicoes_1)
print(' > Acurácia', acc1)
print(' > Precisão', precision_score(rotulos, predicoes_1))
print(' > Recobrimento', recall_score(rotulos, predicoes_1))
print(' > Matriz de confusão:', confusion_matrix(rotulos, predicoes_1))

print('M2')
acc2 = accuracy_score(rotulos, predicoes_2)
print(' > Acurácia', acc2)
print(' > Precisão', precision_score(rotulos, predicoes_2))
print(' > Recobrimento', recall_score(rotulos, predicoes_2))
print(' > Matriz de confusão:', confusion_matrix(rotulos, predicoes_2))

#Obtêm os pesos de cada modelo utilizando a acurácia
acc_total  = acc0 + acc1 + acc2
peso_clf_0 =  acc0/acc_total
peso_clf_1 =  acc1/acc_total
peso_clf_2 =  acc2/acc_total

print('\nPesos')
print(' > Clf 0: ', peso_clf_0)
print(' > Clf 1: ', peso_clf_1)
print(' > Clf 2: ', peso_clf_2)

# Armazena os pesos para uso posterior
pickle.dump(np.array([peso_clf_0, peso_clf_1, peso_clf_2]), open(config.DIR_MODELOS + 'pesos.pkl', 'wb'))

joblib.dump(clf_0, config.DIR_MODELOS + 'clf_0.jl')
joblib.dump(clf_1, config.DIR_MODELOS + 'clf_1.jl')
joblib.dump(clf_2, config.DIR_MODELOS + 'clf_2.jl')
