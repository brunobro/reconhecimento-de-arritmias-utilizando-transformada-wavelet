# -*- coding: utf-8 -*-
'''
Este arquivo tem os principais parâmetros utilizados para a implmentação dos
scripts para Reconhecimento de Arritimas Cardíacas
'''

import numpy as np

'''
Diretórios
'''
DIR_DB        = 'MIT-DB/'
DIR_DB_TESTE  = 'MIT-DB/'
DIR_ATRIBUTOS = 'atributos/'
DIR_MODELOS   = 'modelos/'

'''
MIT/BIH
Registros utilizados para treinamento e validação
'''
REG_TREINO = [101, 102, 104, 106, 108, 109, 112, 114, 115, 116,
              118, 119, 122, 124, 201, 203, 205, 207,
              208, 209, 215, 220, 223, 230, 232, 234]

REG_TESTE  = [100, 103, 105, 107, 111, 113, 117, 121, 123,
              200, 202, 210, 212, 213, 214, 217, 219, 221,
              222, 228, 231, 233]

'''
Quantidade de amostras a serem consideradas antes e após a onda R
para a taxa de amostragem de 360 Hz
'''
ANTES_R = 0.3
APOS_R  = 0.5

'''
Rótulos physionet que são considerados como da classe "Normal"
Os demais são da classe "Abnormal"
'''
rotulos_NORMAL = ['N', 'L', 'R', 'e', 'j']

'''
Rótulos physionet desconsiderados
'''
rotulos_EXCLUIDOS = ['/', 'f', 'Q', 'F', '+'] #+ representa mudança de rítimo

'''
Configurações da CWT
'''
mother_wavelet = 'cmor1.0-1.0'
cwt_scales     = np.arange(10, 100, 1)
vec_interval   = int(len(cwt_scales)/3)
