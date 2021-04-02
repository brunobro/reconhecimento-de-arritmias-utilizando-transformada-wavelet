# -*- coding: utf-8 -*-
'''
1. Extração de Atribuitos: Lê os segmentos dos sinais de ECG, computa a CWT
destes e faz a média para três sequências de intervalos
'''
import wfdb
import pickle
import numpy as np
import pywt
import config

'''
Salva todos os atributos para posterior treinamento e validação
f: treino ou teste
'''
def salva_atributos(regs, f = ''):

    todos_segmentos_QRS_0 = []
    todos_segmentos_QRS_1 = []
    todos_segmentos_QRS_2 = []
    todos_rotulos_QRS     = []

    for reg in regs:
        print(' > ', reg)

        #Para armazenar os segmentos dos ECGs
        segmentos_QRS_0, segmentos_QRS_1, segmentos_QRS_2, rotulos_QRS = obtem_qrs(reg)

        for i in range(0, len(rotulos_QRS)):
            todos_segmentos_QRS_0.append(segmentos_QRS_0[i, :])
            todos_segmentos_QRS_1.append(segmentos_QRS_1[i, :])
            todos_segmentos_QRS_2.append(segmentos_QRS_2[i, :])
            todos_rotulos_QRS.append(rotulos_QRS[i])

    #Converte a lista dos atributos para array
    todos_segmentos_QRS_0 = np.array(todos_segmentos_QRS_0)
    todos_segmentos_QRS_1 = np.array(todos_segmentos_QRS_1)
    todos_segmentos_QRS_2 = np.array(todos_segmentos_QRS_2)

    #Serializa os atributos e os rótulos para uso posterior
    pickle.dump(todos_segmentos_QRS_0, open(config.DIR_ATRIBUTOS + 'atributos_' + f + '_0.pkl', 'wb'))
    pickle.dump(todos_segmentos_QRS_1, open(config.DIR_ATRIBUTOS + 'atributos_' + f + '_1.pkl', 'wb'))
    pickle.dump(todos_segmentos_QRS_2, open(config.DIR_ATRIBUTOS + 'atributos_' + f + '_2.pkl', 'wb'))
    pickle.dump(todos_rotulos_QRS    , open(config.DIR_ATRIBUTOS + 'rotulos_' + f + '.pkl', 'wb'))

    print('\n Total de instâncias: ', todos_segmentos_QRS_0.shape[0])
    print(' > Classe positiva: ', todos_rotulos_QRS.count(1))
    print(' > Classe negativa: ', todos_rotulos_QRS.count(0), '\n\n')

'''
Cria uma lista onde cada elemento tem as amostras dos complexos QRS com 
a arritmia considerada
'''
def obtem_qrs(reg):

    # Fixa uma registro ECG
    rec = config.DIR_DB + str(reg)

    #Lê o sinal de ECG inteiro nas duas derivações disponíveis
    data, info = wfdb.rdsamp(rec)
    ecg = data[:, 0] #Lê apenas a primeira derivação

    #Define a quantidade de amostras antes e depois do pico R
    #multiplicando pela taxa e amostragem do sinal
    AMOSTRAS_ANTES_R  = int(config.ANTES_R * info['fs'])
    AMOSTRAS_APOS_R   = int(config.APOS_R * info['fs'])

    #Define o comprimento da janela (segmento) do sinal que é considerado
    JANELA_ECG = AMOSTRAS_ANTES_R + AMOSTRAS_APOS_R 

    #Lê as anotações do ECG
    ann = wfdb.rdann(rec, 'atr')

    #Quantidade de ondas R (complexos QRS)
    total_QRS = len(ann.sample)

    #Para armazenar os QRS e o rótulo respectivos
    segmentos_QRS_0 = np.zeros((total_QRS, JANELA_ECG))
    segmentos_QRS_1 = np.zeros((total_QRS, JANELA_ECG))
    segmentos_QRS_2 = np.zeros((total_QRS, JANELA_ECG))
    rotulos_QRS     = np.zeros(total_QRS)

    for linha in range(0, total_QRS):

        #Obtêm a localização em amostra do pico R
        onda_R = ann.sample[linha]

        #Obtêm o tipo de anotação para armazenar no arquivo alvo
        tipo_arr = ann.symbol[linha]

        #Insere o ciclo cardiaco e a classe correspondente apenas se o rótulo
        #não for aquele dentre os que devem ser escluídos
        if tipo_arr not in config.rotulos_EXCLUIDOS:

            #Segmento do ciclo cardíaco
            i_i = onda_R - AMOSTRAS_ANTES_R
            i_f = onda_R + AMOSTRAS_APOS_R
            ecg_seg = ecg[i_i:i_f]

            # garante que somente os atributos do tamanho correto sejam inseridos
            # pode ocorrer que os atributos do último ciclo sejam menores, devido ao corte no sinal
            if len(ecg_seg) == JANELA_ECG:

                #Aplica a cwt nas escalas definidas
                coeffs,_ = pywt.cwt(ecg_seg , config.cwt_scales, config.mother_wavelet)
                coeffs = np.abs(np.real(coeffs))**2 #toma o quadrado do valor absoluta da parte real dos coeficientes

                #Adiciona a média dos coeficientes wavelets
                segmentos_QRS_0[linha, :] = np.mean(coeffs[0: config.vec_interval, :])
                segmentos_QRS_1[linha, :] = np.mean(coeffs[config.vec_interval + 1: 2 * config.vec_interval, : ])
                segmentos_QRS_2[linha, :] = np.mean(coeffs[2 * config.vec_interval + 1: 3 * config.vec_interval, : ])

                #Cria uma classe de acordo com os rótulos
                # 0: classe negativa, 1: classe positiva
                if tipo_arr not in config.rotulos_NORMAL:
                    rotulos_QRS[linha] = 1

    return segmentos_QRS_0, segmentos_QRS_1, segmentos_QRS_2, rotulos_QRS


# Cria os atributos de treinamento
print('# Registros para indução')
salva_atributos(config.REG_TREINO, 'inducao')

# Cria os atributos de teste
print('# Registros para validação')
salva_atributos(config.REG_TESTE, 'validacao')

# Mensagem final
print('# Extração de características concluída!')
