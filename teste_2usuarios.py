import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment


# Parâmetros
usuarios = 2
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7460                    # Velocidade orbital em m/s
F_c = 20e9                  # Frequencia de centro em hz
W = 28e6                    # Largura de banda em MHz 28 (28e6 Hz)
T_s = 1e-6                  # Tempo de duração do símbolo em micro segundo
micro = -2.6                # Parâmetro de desvanecimento da chuva em dB*
sigma =  1.6                # Parâmetro de desvanecimento da chuva em dB*
N_0 = 10**(-174/10)         # Densidade espetral do ruído em dBw/Hz para W/Hz
M = 2                       # Número de feixes
g_t = (10**(52.1/10))         # Ganho da antena do satélite em dB p W
g_s = (10**(5/10))          # Lóbulo lateral da antena de satélite em dB p W
m_g_ru = [10**(10/10),
        10**(11/10),
        10**(12/10),
        10**(13/10),
        10**(14/10),
        10**(15/10)]        # Ganho da antena dos usuários, intervalo de 10 a 15 com passo de 1 em dB p W

g_ru = np.random.choice(m_g_ru)

g_b = 10**(10/10)           # Ganho da estação de base em dB p w
P_f = 10**(10/10)           # Potência máxima transmitida em w
P_r = np.full(M, 10**(-111/10))         # Potência de interferência admissível em w
P_c = 10**(10/10)           # Dissipação de potência do circuito em dBw
rho = 0.8                   # Eficiência do amplificador 
R = 6371                    # Raio médio da Terra em Km
xi = 15 * math.pi / 180     # Angulo minimo de elevação dado em graus e convertido para radianos
L_b = 1                     # Perda de transmissão


# Dados de teste

P_T = 10**(20/10)           # Potência total de transmissão em W

coordenadas_lat_long = [(np.float64(-0.04298878668672959), np.float64(-0.044431368204776035)), (np.float64(0.036205613602666685), np.float64(0.22520186856600755))]
distancias = [np.float64(884.5473652217479), np.float64(1722.6211780377405)]
phi_k_list = [np.float64(1.60170331), np.float64 (1.68481879)]
f_k = [np.float64(-15379.265394460484), np.float64(-56623.54795628809)]

##### Equações para algoritmo 3

#Eq.52 (Modelo Global)
def calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M):

    lambda_ = c / F_c  # Comprimento de onda
    L_k = np.zeros(M)
    
    for k in range(M):
        d_m = distancias[k] * 1000  # Converte a distância de km para m
        d_k = P_T * (g_t * g_ru * lambda_**2) / ((4 * np.pi * d_m)**2 * L_b)
        h_k = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)  # Variável aleatória complexa
        L_k[k] = d_k * np.abs(h_k)**2
    
    return L_k

L_k = calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M)
print(f"L_k: {L_k}")


###### Função para calcular f_k
def calcular_fk(v, F_c, c, phi_k_list, M):

    f_k = []
    for k in range(M):
        angle = phi_k_list[k]
        f_k.append((v * F_c / c) * np.cos(angle))
    return f_k

f_k = calcular_fk(v, F_c, c, phi_k_list, M)
print(f"f_k: {f_k}")


#Eq.34 (Modelo Global)
def calculate_pe(P_f, P_T, P_r, g_s, g_b, L_b, M):

    pe1 = P_f
    pe2 = P_T / M
    pe3 = P_r / (g_s * g_b * L_b * M)
    
    # Calcula o valor mínimo entre os três candidatos
    pe0 = np.minimum(pe1, np.minimum(pe2, pe3.min()))
    p_e = np.full(M, pe0)
    
    return p_e

p_e = calculate_pe(P_f, P_T, P_r, g_s, g_b, L_b, M)
print(f"p_e: {p_e}")


#####Eq.21 (Modelo Global)
def calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M):

    I_d = np.zeros(M)

    for k in range(M):
                I_d[k] = p_e[k] * g_t * g_ru * L_k[k] * (1 - np.sinc(f_k[k] * T_s)**2)
    return I_d


#####Eq.20 (Modelo Global)
def calcular_I_i(p_e, g_s, g_ru, L_k, P_f, P_T, P_r, g_b, L_b, M):

    I_i = np.zeros(M)
    
    for k in range(M):
        sum_p_k_prime = p_e[k] * np.sum([1 if k_prime != k else 0 for k_prime in range(M)])
        I_i[k] = g_s * g_ru * L_k[k] * sum_p_k_prime
        
    return I_i


######Eq.27 (Modelo Global)
def f1(p_e, g_t, g_ru, L_k, I_i, N_0, W, f_k, T_s, P_f, P_T, P_r, g_s, g_b, L_b, M):

    I_d = calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M)
    f1_values = np.zeros(M)

    for k in range(M):
        numerator = p_e[k] * g_t * g_ru * L_k[k]
        denominator = I_i[k] + I_d[k] + N_0 * W
        f1_values[k] = np.log2(1 + numerator / denominator)
    
    return f1_values


#######Eq.28 (Modelo Global)
def f2(I_i, I_d, N_0, W, M):

    f2_values = np.zeros(M)

    for k in range(M):
        interference = I_i[k] + I_d[k] + N_0 * W
        f2_values[k] = np.log2(interference)
    
    return f2_values


#Eq.26 (Modelo Global)
def calcular_C_p(f_1, f_2, W, M):
    C_p = np.zeros(M)
    for k in range(M):
        C_p[k] = W * (f_1[k] - f_2[k])
    return C_p


######Eq.25 (Modelo Global)
def grad_f2(I_i, I_d, N_0, W, M):

    f2_values = f2(I_i, I_d, N_0, W, M)
    grad_values = np.gradient(f2_values)
    return grad_values
    

def calculate_tilde_R(p_e, f_1, f_2, gra_f2, p_0, M):
   
    tilde_R = np.zeros(M)
    
    for k in range(M):
        gradient_term = gra_f2[k] * (p_e[k] - p_0[k])
        tilde_R[k] = f_1[k] - (f_2[k] - gradient_term)
    
    return tilde_R


#Eq.24 (Modelo Global)
def tilde_C(tilde_R, W):
    # Somar todas as taxas de transmissão ponderadas pela largura de banda
    tilde_C_p_star = W * np.sum(tilde_R)

    return tilde_C_p_star


#Eq.23 (Modelo Global)
def calcular_D(p_0, P_c, rho):
    D_p_star = P_c + (1 / rho) * sum(p_0)
    return D_p_star



########################################################################################################################################
# Algoritmo 3


# Função objetivo para Dinkelbach
def objetivo_dinkelbach(p_0, p_e, W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, lambda_n, F_c, v, c, phi_k_list, usuarios, M):
    # Calcula a função objetivo do algoritmo de Dinkelbach.
    f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)
    f_1 = f1(p_e, g_t, g_ru, L_k, I_i, N_0, W, f_k, T_s, P_f, P_T, P_r, g_s, g_b, L_b, M)
    f_2 = f2(I_i, I_d, N_0, W, M)
    gra_f2 = grad_f2(I_i, I_d, N_0, W, M)
    tilde_R = calculate_tilde_R(p_e, f_1, f_2, gra_f2, p_0, M)
    tilde_C_p = tilde_C(tilde_R, W)
    D_p = calcular_D(p_0, P_c, rho)
    return -(tilde_C_p - lambda_n * D_p)

# Otimização usando minimize
def resolver_problema_otimizacao_dinkelbach(p_e, lambda_n, p_0, c, P_T, g_t, g_ru, F_c, distancias, L_b, P_f, P_r, g_s, g_b, M, T_s, v, phi_k_list, usuarios, W, N_0, P_c, rho):
    f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)
    L_k = calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M)
    I_i = calcular_I_i(p_e, g_s, g_ru, L_k, P_f, P_T, P_r, g_b, L_b, M)
    I_d = calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M)

# Definição das restrições usando lambda
    constraints = [
        {'type': 'eq', 'fun': lambda p_0: p_0[0] - 10},  # Fixar o primeiro usuário em 10W
        {'type': 'ineq', 'fun': lambda p_0: P_T - sum(p_0)},  # Limite superior total de potência
        {'type': 'ineq', 'fun': lambda p_0: P_f - np.array(p_0)},  # Limite superior para cada potência individual
        {'type': 'ineq', 'fun': lambda p_0: sum(p_0) - (P_r[0] / (g_s * g_b * L_b)).sum()}  # Limite inferior para a soma das potências
        ]


    result = minimize(
        objetivo_dinkelbach,
        p_0,
        args=(p_e, W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, lambda_n, F_c, v, c, phi_k_list, usuarios, M),
        constraints=constraints,
        method='trust-constr',
        bounds=[(p_e[0], P_T) for _ in p_0],
        options={'disp': True} #Mostra informações sobre o progresso da minimização.
    )
    return result

# Algoritmo de Dinkelbach
def dinkelbach_algorithm(p_0, p_e, c, P_T, g_t, g_ru, F_c, distancias, L_b, P_f, P_r, g_s, g_b, M, T_s, v, phi_k_list, usuarios, P_c, rho, N_0, W, epsilon=1e-5):
    # Inicializa o parâmetro lambda
    lambda_n = 0
    n = 0  # Contador de iterações

    while True:
        # Resolve o problema de otimização com o valor atual de lambda_n
        result = resolver_problema_otimizacao_dinkelbach(p_e, lambda_n, p_0, c, P_T, g_t, g_ru, F_c, distancias, L_b, P_f, P_r, g_s, g_b, M, T_s, v, phi_k_list, usuarios, W, N_0, P_c, rho)
        p_star = result.x  # Potências ótimas encontradas

        f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)
        L_k = calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M)
        I_i = calcular_I_i(p_e, g_s, g_ru, L_k, P_f, P_T, P_r, g_b, L_b, M)
        I_d = calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M)

        f_1 = f1(p_e, g_t, g_ru, L_k, I_i, N_0, W, f_k, T_s, P_f, P_T, P_r, g_s, g_b, L_b, M)
        f_2 = f2(I_i, I_d, N_0, W, M)
        gra_f2 = grad_f2(I_i, I_d, N_0, W, M)
        tilde_R = calculate_tilde_R(p_e, f_1, f_2, gra_f2, p_0, M)
        tilde_C_p_star = tilde_C(tilde_R, W)
        D_p_star = calcular_D(p_0, P_c, rho)

        # Calcula F(lambda_n)
        F_lambda_n = tilde_C_p_star - lambda_n * D_p_star

        # Verifica a condição de parada
        if F_lambda_n < epsilon:
            break

        # Atualiza lambda_n e p_0 para a próxima iteração
        lambda_n = tilde_C_p_star / D_p_star
        p_0 = p_star
        n += 1

    return p_0, lambda_n




# Equações para algoritmo 2

# Eq.12 (Modelo Global)
# Função para calcular p_km para cada subportadora e usuário
def calcular_p_km(P_T, P_f, P_r, g_s, g_b, L_b, M):
    # Calcular os três valores possíveis para P_eq
    P_eq_1 = P_f
    P_eq_2 = P_T / M
    P_eq_3 = P_r / (g_s * g_b * L_b * M)

    # Encontrar o valor mínimo entre os três valores possíveis
    P_eq = np.minimum(P_eq_1, np.minimum(P_eq_2, P_eq_3.min()))
    
    # Criar a matriz p_km com o valor P_eq
    p_km = np.full((M, M), P_eq)
    return p_km



# Eq.15 (Modelo Global) Algoritmo 2
# Função para calcular eta(X)
def calcular_eta(M, p_km, P_c, P_r, rho, W, g_t, g_ru, L_k, I_i, I_d, N_0, X):
    numerator = 0
    for k in range(M):
        for m in range(M):
            numerator += W * np.log2(1 + (P_r[m] * g_t * g_ru * L_k[k]) / (I_i[k] + I_d[m] + N_0 * W)) * X[k, m]
    denominator = P_c + (1 / rho) * np.sum(P_r)
    eta = numerator / denominator
    return eta


# Eq.16 (Modelo Global), interferência entre feixes
def calcular_I_ki(g_s, g_ru, L_k, p_km, x_km, M):
    I_ki = np.zeros((M, M))
    for k in range(M):
        for m in range(M):
            interferencia = g_s * g_ru* L_k[k] * np.sum(p_km[:, m] * (1 - x_km[:, m]))
            I_ki[k, m] = interferencia
    return I_ki


# Eq.17 (Modelo Global)
def calcular_q_km(W, P_c, rho, p_km, g_t, g_ru, L_k, I_ki, I_d, N_0):
    q_km = np.zeros((M, M))
    denominador = P_c + (1 / rho) * np.sum(p_km)
    for k in range(M):
        for m in range(M):
            numerador = W * np.log2(1 + (p_km[k, m] * g_t * g_ru * L_k[k]) / (I_ki[k, m] + I_d[m] + N_0 * W))
            q_km[k, m] = numerador / denominador
    return q_km


###### Eq.18 (Modelo Global)
def otimizar_X(q_km, M):
    # Convertendo o problema de maximização para minimização
    cost_matrix = -q_km
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  
    X = np.zeros((M, M))
    X[row_ind, col_ind] = 1
    return X


##### Função iterativa para otimização conjunta
def otimizar_interativamente(p_e, P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, distancias, N_0, max_iter=10, tol=1e-5):
    L_k = calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M)
    f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)
    I_d = calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M) #Interferência Doppler
    I_i = calcular_I_i(p_e, g_s, g_ru, L_k, P_f, P_T, P_r, g_b, L_b, M)

    X = np.zeros((M, M))
    for k in range(M):
        X[k, k % M] = 1

    for iteration in range(max_iter):
        # Calcular p_km
        p_km = calcular_p_km(P_T, P_f, P_r, g_s, g_b, L_b, M)

        # Calcular I_ki
        I_ki = calcular_I_ki(g_s, g_ru, L_k, p_km, X, M)

        # Calcular q_km
        q_km = calcular_q_km(W, P_c, rho, p_km, g_t, g_ru, L_k, I_ki, I_d, N_0)

        # Calcular eta(X)
        eta = calcular_eta(M, p_km, P_c, P_r, rho, W, g_t, g_ru, L_k, I_i, I_d, N_0, X)
        print(f'Iteração {iteration}: eta = {eta}')

        # Otimizar X
        X_new = otimizar_X(q_km, M)

        # Verificar convergência
        if np.allclose(X, X_new, atol=tol):
            print(f'Converged after {iteration} iterations')
            break

        X = X_new

    return X, p_km, I_ki, q_km



####################################################################################
# Equações para algoritmo 1



def sinc(x):
    #Implementação da função sinc.
    return np.sinc(x / np.pi)  # A função np.sinc(x) é normalizada como sin(pi*x)/(pi*x)

def calculate_interuser_interference(p_km, g_t, g_ru, L_k, f_k, T_s):
   
    K, M = p_km.shape  # K é o número de usuários, M é o número de potências de transmissão
    I_d_km = np.zeros((K, M))  # Inicializar a matriz de interferência interusuário com zeros
    
    for k in range(K):
        for m in range(M):
            sinc_term = sinc(f_k[k] * T_s)
            I_d_km[k, m] = p_km[k, m] * g_t * g_ru * L_k[k] * (1 - sinc_term**2)

    return I_d_km


# Eq.3 (Modelo Global)
def calculate_intrauser_interference(p_km, g_s, g_ru, L_k, X):

    K, M = p_km.shape  # K é o número de usuários, M é o número de potências de transmissão
    I_i_km = np.zeros((K, M))  # Inicializar a matriz de interferência intrausuário com zeros
    
    for k in range(K):
        for m in range(M):
            interference_sum = 0
            for k_prime in range(K):
                for m_prime in range(M):
                    if k_prime != k or m_prime != m:
                        interference_sum += p_km[k_prime, m_prime] * X[k_prime, m_prime]
            I_i_km[k, m] = g_s * g_ru * L_k[k] * interference_sum

    return I_i_km


##### Eq.2 (Modelo Global)
def calculate_snr(p_km, g_t, g_ru, L_k, I_i_km, I_d_km, N_0, W):

    # Certificar que as dimensões das entradas são compatíveis
    K, M = p_km.shape  # K é o número de usuários, M é o número de potências de transmissão
    
    # Inicializar a matriz de SNR com zeros
    gamma_km = np.zeros((K, M))
    
    # Calcular a SNR para cada par (k, m)
    for k in range(K):
        for m in range(M):
            numerator = p_km[k, m] * g_t * g_ru * L_k[k]
            denominator = I_i_km[k, m] + I_d_km[k, m] + N_0 * W
            gamma_km[k, m] = numerator / denominator
    
    return gamma_km


# Eq.6 (Modelo Global)
def calculate_sum_rate(W, gamma_km, X):

    K, M = gamma_km.shape
    R_k = np.zeros(K)
    for k in range(K):
        for m in range(M):
            if X[k, m] == 1:
                R_k[k] += W * np.log2(1 + gamma_km[k, m])
    return R_k


##### Eq.10 (Modelo Global)
def eta_ef(X, p_km, P_c, rho, W, gamma_km):
    R_k = calculate_sum_rate(W, gamma_km, X)
    total_rate = np.sum(R_k)
    total_power = P_c + (1 / rho) * np.sum(p_km * X)
    return total_rate / total_power



###### Algoritmo 1 ######



def algoritmo_1(P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, c, F_c, T_s, v, phi_k_list, usuarios, N_0, epsilon=1e-6):

    # Inicializações
    p_eq = np.min([P_f, P_T / M, np.min(P_r / (g_s * g_b * L_b * M))])
    p_0 = np.full(M, p_eq)
    i = 0

    L_k = calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M)
    f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)

    # Listas para armazenar a evolução da potência e eficiência energética
    evolucao_potencia = []
    evolucao_eficiencia_energetica = []

        # Função para imprimir os valores iniciais e verificar restrições
    def imprimir_verificacao_inicial(p_0, P_T, P_f, P_r, g_s, g_b, L_b):
        print("Valores iniciais de p_0:", p_0)
        print("Soma de p_0:", sum(p_0))
        print("Limite superior de potência total (P_T):", P_T)
        print("Limite superior para cada potência individual (P_f):", P_f)
        print("Limite inferior para soma de potências:", p_e)
        print("Verificação de cada restrição:")
        print("1. p_0[0] == 10:", p_0[0] == 10)
        print("2. sum(p_0) <= P_T:", sum(p_0) <= P_T)
        print("3. Todos p_0 <= P_f:", all(p <= P_f for p in p_0))
        print("4. sum(p_0) >= (P_r / (g_s * g_b * L_b)):", sum(p_0) >= p_e)

    p_0_inicial = [10] + [1] * (len(p_0) - 1)  # Ajuste inicial de p_0 conforme necessário
    imprimir_verificacao_inicial(p_0_inicial, P_T, P_f, P_r, g_s, g_b, L_b)

    while True:
        # Algoritmo 2: Beam assignment
        X, p_km, I_ki, q_km = otimizar_interativamente(p_e, P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, distancias, N_0, max_iter=100, tol=1e-6)

        # Algoritmo 3: Alocação de potência
        p_star, lambda_n = dinkelbach_algorithm(p_0, p_e, c, P_T, g_t, g_ru, F_c, distancias, L_b, P_f, P_r, g_s, g_b, M, T_s, v, phi_k_list, usuarios, P_c, rho, N_0, W, epsilon=1e-6)

        # Critério de parada
        I_i_km = calculate_intrauser_interference(p_km, g_s, g_ru, L_k, X)
        I_d_km = calculate_interuser_interference(p_km, g_t, g_ru, L_k, f_k, T_s)
        gamma_km = calculate_snr(p_km, g_t, g_ru, L_k, I_i_km, I_d_km, N_0, W)
        eficiencia_energetica_atual = eta_ef(X, p_star, P_c, rho, W, gamma_km)

        if i > 0 and (evolucao_eficiencia_energetica[-1] - eficiencia_energetica_atual) < epsilon:
            break
        
        # Armazenando a evolução da potência e da eficiência energética
        evolucao_potencia.append(p_star)
        evolucao_eficiencia_energetica.append(eficiencia_energetica_atual)

        # Atualização para a próxima iteração
        p_0 = p_star
        i += 1

    return p_star, X, evolucao_potencia, evolucao_eficiencia_energetica

# Executando o algoritmo 1 com os valores definidos
p_star, X_star, evolucao_potencia, evolucao_eficiencia_energetica = algoritmo_1(P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, c, F_c, T_s, v, phi_k_list, usuarios, N_0, epsilon=1e-6)

print(f"p*:\n {p_star}")
print(f"X*:\n {X_star}")

# Evolução da potência de cada feixe
for m in range(len(evolucao_potencia[0])):
    plt.plot(range(len(evolucao_potencia)), [p[m] for p in evolucao_potencia], label=f'Feixe {m+1}')
plt.xlabel('Iteração')
plt.ylabel('Potência (W)')
plt.legend()
plt.grid(True)
plt.show()

# Evolução da eficiência energética em função da potência total
plt.plot(evolucao_eficiencia_energetica, marker='o', color='b')
plt.xlabel("Iteração")
plt.ylabel("Eficiência Energética")
plt.grid(True)
plt.title("Evolução da Eficiência Energética em Função da Potência")
plt.show()