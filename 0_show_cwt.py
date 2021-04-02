# -*- coding: utf-8 -*-
'''
Este script é utilizado apenas para gerar um scalograma de exemplo
'''

import pywt
import wfdb
import numpy as np
import matplotlib.pyplot as plt

#MIT-DB directory
dir = 'MIT-DB/'

#Lê um trecho (3 segundos) do sinal de ECG do registro 233
Fs          = 360
time_seg    = 3
ECG, fields = wfdb.rdsamp(dir + '233', sampfrom=0, sampto=time_seg*Fs, channels=[1])

#Mostra a relação escala frequência
wavelet     = 'cmor1.0-1.0'
dt          = 1/Fs
step_scales = 5
min_scales  = 10
max_scales  = 100
scales      = np.arange(min_scales , max_scales + step_scales, step_scales)
frequencies = pywt.scale2frequency(wavelet , scales) / dt
'''
print('Escala','/','Frequência')
for k, f in enumerate(frequencies):
    print(scales[k], '/', f)
'''
#Aplica a CWT ao sinal de ECG
coeffs,_ = pywt.cwt(ECG, scales, wavelet)

e_coeffs = np.abs(np.real(coeffs))**2
e_coeffs = e_coeffs.reshape(e_coeffs.shape[0], e_coeffs.shape[1])

t = np.arange(0, time_seg, 1/Fs)

plt.figure(dpi=150)
plt.subplot(211)
plt.plot(t, ECG)
plt.xlabel('Tempo (segundos)')
plt.ylabel('Amplitude (mV)')
plt.title('(a)')
plt.grid()
plt.xlim((0, time_seg))
plt.subplot(212)
plt.imshow(e_coeffs, extent=[0, time_seg, max_scales , min_scales ], cmap='jet', aspect='auto',
           vmax=e_coeffs.max(), vmin=e_coeffs.min())
plt.xlabel('Tempo (segundos)')
plt.ylabel('Escala ($s$)')
plt.yticks(np.arange(min_scales, max_scales + 10, 10))
plt.title('(b)')
plt.tight_layout()
plt.show()






