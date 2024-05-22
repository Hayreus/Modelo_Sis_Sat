import math
import numpy as np
import matplotlib as plt
from scipy.optimize import minimize
from munkres import Munkres, print_matrix

# Parâmetros
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7.46                    # Velocidade orbital em km/s
F_c = 20                    # Frequencia de centro em Ghz
W = 28                      # Largura de banda em MHz (28e6)
T_s = 1                     # Tempo de duração do símbolo em micro segundo
micro = -2.6                # Parâmetro de desvanecimento da chuva em dB*
sigma =  1.6                # Parâmetro de desvanecimento da chuva em dB*
N_0 = -172                  # Densidade espetral do ruído em dBw/Hz
M = 7                       # Número de feixes de antenas
g_t = 52.1                  # Ganho da antena do satélite em dB
g_s = 5                     # Lóbulo lateral da antena de satélite em dB
g_k = range(10, 16, 1)      # Ganho da antena dos usuários, intervalo de 10 a 15 com passo de 0.5 em dB
g_b = 5                     # Ganho da estação de base em dB
P_f = 0                     # Potência máxima transmitida em dBw
P_r = -111                  # potência de interferência admissível em dBw
P_c = 10                    # Dissipação de potência do circuito em dBw
rho = 0.8 

# Dados de teste
p = [0.5, 1.0, 1.5, 2.0, 2.5]  # Potência transmitida em cada feixe
g_ru = [12, 14, 13, 15, 11]  # Ganho da antena dos usuários em dB
L = [1e-3, 2e-3, 1.5e-3, 1e-3, 2e-3]  # Atenuação de percurso para cada feixe

I_d = [1e-10, 2e-10, 1.5e-10, 1e-10, 2e-10]  # Interferência de outras fontes
N_0 = 10**(-172/10)  # Densidade espectral do ruído convertida de dBw/Hz para W/Hz

#Eq.21
def calcular_I_d(p, g_t, g_ru, L, f_k, T_s):
    I_d = [0] * len(p)

    for k in range(len(p)):
        # Calcula o termo sinc(f_k * T_s)
        sinc_term = math.sin(math.pi * f_k[k] * T_s) / (math.pi * f_k[k] * T_s) if f_k[k] * T_s != 0 else 1
        
        # Calcula a interferência externa para o feixe k
        I_d[k] = p[k] * g_t * g_ru[k] * L[k] * (1 - sinc_term**2)
    
    return I_d

I_d = calcular_I_d(p, g_t, g_ru, L,)

#Eq.20
def calcular_I_i(p, g_s, g_ru, L):
    I_i = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula a interferência interna para o feixe k
        I_i[k] = g_s * g_ru[k] * L[k] * sum(p[k_prime] for k_prime in range(len(p)) if k_prime != k)
    
    return I_i

I_i = calcular_I_i(p, g_s, g_ru, L)
print(f"Lista de interferências internas para cada feixe: {I_i}")

#Eq.19
def calcular_eta(p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0):

    K = len(p)  # Número de feixes
    
    # Calcula o numerador da eficiência energética
    C_p = 0
    for k in range(K):
        numerator = p[k] * g_t * g_ru[k] * L[k]
        denominator = I_i[k] + I_d[k] + N_0 * W
        C_p += W * math.log2(1 + numerator / denominator)
    
    # Calcula o denominador da eficiência energética
    D_p = P_c + (1 / rho) * sum(p)
    
    # Calcula a eficiência energética como a razão entre o numerador e o denominador
    eta = C_p / D_p
    
    return eta


# Chama a função a ser testada
eta = calcular_eta(p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0)

# Verifica o resultado esperado
print(f"Eficiência Energética Calculada (W): {eta}")
