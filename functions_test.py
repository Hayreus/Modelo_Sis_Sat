import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from munkres import Munkres, print_matrix

# Parâmetros
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7.46e3                  # Velocidade orbital em m/s
F_c = 20                  # Frequencia de centro em Ghz
W = 28e6                    # Largura de banda em MHz 28 (28e6 Hz)
T_s = 1e-6                  # Tempo de duração do símbolo em micro segundo
N_0 = 10**(-172/10)         # Densidade espetral do ruído em dBw/Hz para W/Hz
M = 7                       # Número de feixes de antenas
g_t = 52.1                  # Ganho da antena do satélite em dB
g_s = 5                     # Lóbulo lateral da antena de satélite em dB
g_k = [10, 11, 12, 13, 14, 15]      # Ganho da antena dos usuários, intervalo de 10 a 15.
g_b = 5                     # Ganho da estação de base em dB
P_f = 0                     # Potência máxima transmitida em dBw***
P_r = 10                    # Potência de interferência admissível em dBw
P_c = 10                    # Dissipação de potência do circuito em dBw
rho = 0.8                   # Eficiência do amplificador 
R = 6371                    # Raio médio da Terra em Km
xi = 15 * math.pi / 180     # Angulo minimo de elevação dado em graus e convertido para radianos
n =  7                      # Número de celulas hexagonais
delta = 0.35                # Ganho do lóbulo lateral
theta = 90 * math.pi / 180  # Largura do feixe da antena de 0 a 360
micro = -2.6             # Parâmetro de desvanecimento da chuva em dB*
sigma =  1.6             # Parâmetro de desvanecimento da chuva em dB*



# Dados de teste
phi = np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])     # Vetor phi com 7 posições
L_b = 1                     # Perda de transmissão (ajuste conforme necessário)
P_T = 10**(30/10)                    # Potência total de transmissão em dBw
p = [0.5, 1.0, 1.5, 2.0, 2.5, 1.7, 1.3]     # Potência transmitida em cada feixe
g_ru = [12, 14, 13, 15, 11, 10, 16]         # Ganho da antena dos usuários em dB
L = [1e-3, 2e-3, 1.5e-3, 1e-3, 2e-3, 2e-3, 1.7e-3]      # Atenuação de percurso para cada feixe
x = np.array([[1, 0, 0, 0, 0, 0, 0], 
              [0, 1, 0, 0, 0, 0, 0], 
              [0, 0, 1, 0, 0, 0, 0], 
              [0, 0, 0, 1, 0, 0, 0], 
              [0, 0, 0, 0, 1, 0, 0], 
              [0, 0, 0, 0, 0, 1, 0], 
              [0, 0, 0, 0, 0, 0, 1]])        # É uma matriz variável binária que indica a atribuição do feixe.



#Eq. 1 (Posição Global)
def calcular_psi(R, h, xi):

    # Cálculo do termo dentro do arco seno
    termo_arco_seno = R / (R + h) * math.cos(xi)
    
    # Arco seno
    arco_seno = math.asin(termo_arco_seno)
    
    # Cálculo do ângulo de cobertura area
    area = 2 * R * (math.pi / 2 - xi - arco_seno)
    
    psi = area / R
    return psi
psi = calcular_psi(R, h, xi)
print (f"Ângulo de cobertura em radianos: {psi}")


#Eq. 2 (Posição Global)
def calcular_beta(psi, n):
    beta = (2 * psi) / (2 * n + 1)
    
    return beta
beta = calcular_beta(psi, n)
print(f"Angulo de cobertura de cada célula: {beta}")


#Eq. 3 (Posição Global)
def calcular_Nc(n):
    Nc = 1 + (6 * n * (n + 1)) / 2
    
    return Nc
Nc = calcular_Nc(n)
print(f"Número de feixes pontuais: {Nc}")


#Eq. 4 (Posição Global)
def calcular_theta_0(R, h, beta):
    numerador = R * math.sin(beta / 2)
    denominador = h + R - R * math.cos(beta / 2)
    theta_0 = (math.atan2(numerador, denominador))/2
    
    return theta_0
theta_0 = calcular_theta_0(R, h, beta)
print(f"largura do feixe da célula central: {theta_0}")


#Eq. 5 (Posição Global)
def calcular_theta_n(R, h, beta, Nc, theta_0, n):

    theta_n = np.zeros(n)
    for i in range(n):
        theta_n[i] = calcular_theta_n_individual(R, h, beta, Nc, theta_0, i+1)
    return theta_n

def calcular_theta_n_individual(R, h, beta, Nc, theta_0, n):

    theta_k_sum = 0
    for k in range(1, n):
        theta_k_sum += calcular_theta_k(R, h, beta, Nc, theta_0, k)
    theta_n = math.atan((R * math.sin((2 * n + 1) * beta / 2)) / (h + R - R * math.cos((2 * n + 1) * beta / 2))) - theta_k_sum - (theta_0 / 2)
    return theta_n

def calcular_theta_k(R, h, beta, Nc, theta_0, n):

    k = n
    return Nc * theta_0 * (R / (h + R))**k * math.sin(theta_0 / 2) / (2 * k * math.sin(beta / 2))

theta_n = calcular_theta_n(R, h, beta, Nc, theta_0, n)
print(f"theta_n: {theta_n}")



############################################################################################



#Eq. 5 (Modelo Global)
def calcular_fk(v, F_c, c, theta_n, p):
    K = len(p)
    f_k = []
    for k in range(K):
        angle = theta_n[k]
        f_k.append((v * F_c / c) * np.cos(angle))
    return f_k

f_k = calcular_fk(v, F_c, c, theta_n, p)
print(f"Frequência desviada associada ao k-ésimo usuário: {f_k}")


#Eq.21 (Modelo Global)
def calcular_I_d(p, g_t, g_ru, L, f_k, T_s):
    I_d = [0] * len(p)

    for k in range(len(p)):
        # Calcula o termo sinc(f_k * T_s)
        sinc_term = math.sin(math.pi * f_k[k] * T_s) / (math.pi * f_k[k] * T_s) if f_k[k] * T_s != 0 else 1
        
        # Calcula a interferência externa para o feixe k
        I_d[k] = p[k] * g_t * g_ru[k] * L[k] * (1 - sinc_term**2)
    
    return I_d
I_d = calcular_I_d(p, g_t, g_ru, L, f_k, T_s)
print(f"Interferência de outras fontes: {I_d}")


#Eq.20 (Modelo Global)
def calcular_I_i(p, g_s, g_ru, L):
    I_i = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula a interferência interna para o feixe k
        I_i[k] = g_s * g_ru[k] * L[k] * sum(p[k_prime] for k_prime in range(len(p)) if k_prime != k)
    
    return I_i
I_i = calcular_I_i(p, g_s, g_ru, L)
print(f"Lista de interferências internas para cada feixe: {I_i}")


#Eq.19 (Modelo Global)
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
eta = calcular_eta(p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0)
print(f"Eficiência Energética Calculada (W): {eta}")


#Eq. 22 (Modelo Global)
def objective(p, W, g_t, g_ru, L, I_i, I_d, N0, P_c, rho):
    num = np.sum([W * np.log2(1 + (p[k] * g_t * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N0 * W)) for k in range(len(p))])
    denom = P_c + (1 / rho) * np.sum(p)
    return -num / denom  # Negativo porque estamos maximizando

# Restrições
def constraint1(p, P_T):
    return P_T - np.sum(p)

def constraint2(p, P_f):
    return P_f - np.max(p)

def constraint3(p, P_r, g_s, g_b, L_b):
    return P_r - np.sum(p) * g_s * g_b * L_b

# Número de feixes ativos
A = len(g_ru)

# Inicialização das potências (valores iniciais)
p_0 = np.ones(A) * (P_T / A)

# Definição das restrições
con1 = {'type': 'ineq', 'fun': constraint1, 'args': (P_T,)}
con2 = {'type': 'ineq', 'fun': constraint2, 'args': (P_f,)}
con3 = {'type': 'ineq', 'fun': constraint3, 'args': (P_r, g_s, g_b, L_b)}
cons = [con1, con2, con3]

# Limites para as potências
bounds = [(0, P_f) for _ in range(A)]

# Resolução do problema de otimização
solution = minimize(objective, p_0, args=(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho),
                    method='SLSQP', bounds=bounds, constraints=cons)

# Potências ótimas
p_max = -solution.fun
p_star = solution.x

print("Potências ótimas dos feixes:", p_star)
print("Valor máximo da eficiência energética:", p_max)


#Eq.27 (Modelo Global)
def f1(p, g_t, g_ru, L, I_i, I_d, N_0, W):
    f1_values = np.zeros(len(p))
    
    for k in range(len(p)):
        numerator = p[k] * g_t * g_ru[k] * L[k]
        denominator = I_i[k] + I_d[k] + N_0 * W
        f1_values[k] = np.log2(1 + numerator / denominator)
    
    return f1_values


#Eq.28 (Modelo Global)
def f2(I_i, I_d, N_0, W):
    f2_values = np.zeros(len(I_i))
    
    for k in range(len(I_i)):
        interference = I_i[k] + I_d[k] + N_0 * W
        f2_values[k] = np.log2(interference)
    
    return f2_values
f_1 = f1(p, g_t, g_ru, L, I_i, I_d, N_0, W)
f_2 = f2(I_i, I_d, N_0, W)

print(f"Valor de f1: {f_1}")
print(f"Valor de f2: {f_2}")


#Eq.26 (Modelo Global)
def calcular_C_p(p, f_1, f_2, W):
    C_p = np.zeros(len(p))
    for k in range(len(p)):
        C_p[k] = W * (f_1[k] - f_2[k])
    return C_p
S_p = calcular_C_p(p, f_1, f_2, W)
print(f"soma ponderada das diferenças entre f1(p_k) e f2(p_k): {S_p}")


#Eq.25 (Modelo Global)
def grad_f2(I_i, I_d, N_0, W):
    K = len(I_i)  
    grad_values = np.zeros(K)
    
    for k in range(K):
        interference = I_i[k] + I_d[k] + N_0 * W
        grad_values[k] = 1 / (np.log(2) * interference)  # Derivada de log2(interference)
    
    return grad_values

gra_f2 = grad_f2(I_i, I_d, N_0, W)
print(f"Gradiente de f2: {gra_f2}")


def calculate_tilde_R(gra_f2, f_1, f_2, p_0, p):
   
    K = len(p)  # Número de feixes
    tilde_R = np.zeros(K)
    
    for k in range(K):
        gradient_term = gra_f2[k] * (p[k] - p_0[k])
        tilde_R[k] = f_1[k] - (f_2[k] - gradient_term)
    
    return tilde_R

tilde_R = calculate_tilde_R(gra_f2, f_1, f_2, p_0, p)
print(f"Limite inferior da taxa de soma do utilizador k: {tilde_R}")