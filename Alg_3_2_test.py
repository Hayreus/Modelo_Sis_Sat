import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from munkres import Munkres, print_matrix

# Parâmetros
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7.46                    # Velocidade orbital em km/s
F_c = 20                    # Frequencia de centro em Ghz
W = 28 * 10**6               # Largura de banda em MHz 28 (28e6 Hz)
T_s = 1                     # Tempo de duração do símbolo em micro segundo
micro = -2.6                # Parâmetro de desvanecimento da chuva em dB*
sigma =  1.6                # Parâmetro de desvanecimento da chuva em dB*
N_0 = 10**(-172/10)   # Densidade espetral do ruído em dBw/Hz para W/Hz
M = 7                       # Número de feixes de antenas
g_t = 52.1                  # Ganho da antena do satélite em dB
g_s = 5                     # Lóbulo lateral da antena de satélite em dB
g_k = [10, 11, 12, 13, 14, 15]      # Ganho da antena dos usuários, intervalo de 10 a 15 com passo de 0.5 em dB
g_b = 5                     # Ganho da estação de base em dB
P_f = 10                    # Potência máxima transmitida em dBw***
P_r = -111                  # Potência de interferência admissível em dBw
P_c = 10                    # Dissipação de potência do circuito em dBw
rho = 0.8                   # Eficiência do amplificador 
R = 6371                    # Raio médio da Terra em Km
xi = 15 * math.pi / 180     # Angulo minimo de elevação dado em graus e convertido para radianos
n =  7                     # Número de celulas hexagonais


# Dados de teste
L_b = 1                 # Perda de transmissão (ajuste conforme necessário)
P_T = 30                   # Potência total de transmissão em dBw
p = [0.5, 1.0, 1.5, 2.0, 2.5, 1.7, 1.3]     # Potência transmitida em cada feixe
g_ru = [12, 14, 13, 15, 11, 10, 16]         # Ganho da antena dos usuários em dB
L = [1e-3, 2e-3, 1.5e-3, 1e-3, 2e-3, 2e-3, 1.7e-3]  # Atenuação de percurso para cada feixe


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
print (f"--Ângulo de cobertura em radianos: {psi}")


#Eq. 2 (Posição Global)
def calcular_beta(psi, n):
    beta = (2 * psi) / (2 * n + 1)
    
    return beta
beta = calcular_beta(psi, n)
print(f"--Angulo de cobertura de cada célula: {beta}")


#Eq. 3 (Posição Global)
def calcular_Nc(n):
    Nc = 1 + (6 * n * (n + 1)) / 2
    
    return Nc
Nc = calcular_Nc(n)
print(f"--Número de feixes pontuais: {Nc}")


#Eq. 4 (Posição Global)
def calcular_theta_0(R, h, beta):
    numerador = R * math.sin(beta / 2)
    denominador = h + R - R * math.cos(beta / 2)
    theta_0 = (math.atan2(numerador, denominador))/2
    
    return theta_0
theta_0 = calcular_theta_0(R, h, beta)
print(f"--largura do feixe da célula central: {theta_0}")


#Eq. 5 (Posição Global)
def calcular_theta_k(R, h, beta, Nc, theta_0, n):

    k = n
    return Nc * theta_0 * (R / (h + R))**k * math.sin(theta_0 / 2) / (2 * k * math.sin(beta / 2))

def calcular_theta_n_individual(R, h, beta, Nc, theta_0, n):

    theta_k_sum = 0
    for k in range(1, n):
        theta_k_sum += calcular_theta_k(R, h, beta, Nc, theta_0, k)
    theta_n = math.atan((R * math.sin((2 * n + 1) * beta / 2)) / (h + R - R * math.cos((2 * n + 1) * beta / 2))) - theta_k_sum - (theta_0) # A útima parcela seria theta/2, mas ela já está dividido na função anterior
    return theta_n

def calcular_theta_n(R, h, beta, Nc, theta_0, n):

    theta_n = np.zeros(n)
    for i in range(n):
        theta_n[i] = calcular_theta_n_individual(R, h, beta, Nc, theta_0, i+1)
    return theta_n
theta_n = calcular_theta_n(R, h, beta, Nc, theta_0, n)
print("--Largura do feixe da enésima coroa para cada valor de n:", theta_n, "radianos")



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
print(f"--Frequência desviada associada ao k-ésimo usuário: {f_k}")


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
print(f"--Interferência de outras fontes: {I_d}")


#Eq.20 (Modelo Global)
def calcular_I_i(p, g_s, g_ru, L):
    I_i = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula a interferência interna para o feixe k
        I_i[k] = g_s * g_ru[k] * L[k] * sum(p[k_prime] for k_prime in range(len(p)) if k_prime != k)
    
    return I_i
I_i = calcular_I_i(p, g_s, g_ru, L)
print(f"--Lista de interferências internas para cada feixe: {I_i}")


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
print(f"--Eficiência Energética Calculada (W): {eta}")


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

print("--Potências ótimas dos feixes:", p_star)
print("--Valor máximo da eficiência energética:", p_max)


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

print(f"--Valor de f1: {f_1}")
print(f"--Valor de f2: {f_2}")


#Eq.26 (Modelo Global)
def calcular_C_p(p, f_1, f_2, W):
    C_p = np.zeros(len(p))
    for k in range(len(p)):
        C_p[k] = W * (f_1[k] - f_2[k])
    return C_p
S_p = calcular_C_p(p, f_1, f_2, W)
print(f"--Soma ponderada das diferenças entre f1(p_k) e f2(p_k): {S_p}")


#Eq.25 (Modelo Global)
def grad_f2(I_i, I_d, N_0, W):
    K = len(I_i)  # Número de feixes
    grad_values = np.zeros(K)
    
    for k in range(K):
        interference = I_i[k] + I_d[k] + N_0 * W
        grad_values[k] = 1 / (np.log(2) * interference)  # Derivada de log2(interference)
    
    return grad_values
gra_f2 = grad_f2(I_i, I_d, N_0, W)
print(f"--Gradiente de f2: {grad_f2}")

def calculate_tilde_R(p, p_0, g_t, g_ru, L, I_i, I_d, N_0, W):
   
    K = len(p)  # Número de feixes
    tilde_R = np.zeros(K)
    
    for k in range(K):
        gradient_term = gra_f2[k] * (p[k] - p_0[k])
        tilde_R[k] = f_1[k] - (f_2[k] - gradient_term)
    
    return tilde_R

tilde_R = calculate_tilde_R(p, p_0, g_t, g_ru, L, I_i, I_d, N_0, W)
print(f"--Limite inferior da taxa de soma do utilizador k: {tilde_R}")


#Eq.24 (Modelo Global)
def tilde_C(p_star, g_t, g_ru, L, I_i, I_d, N_0, W, p_0):

    # Calcular as taxas de transmissão \tilde{R}_{k}(\mathbf{p}^{*}) para todos os feixes k
    tilde_R_values = calculate_tilde_R(p, p_0, g_t, g_ru, L, I_i, I_d, N_0, W)

    # Somar todas as taxas de transmissão ponderadas pela largura de banda
    tilde_C_p_star = W * np.sum(tilde_R_values)

    return tilde_C_p_star

tilde_C_p_star = tilde_C(p_star, g_t, g_ru, L, I_i, I_d, N_0, W, p_0)
print(f"--Capacidade de transmissão total no ponto ótimo: {tilde_C_p_star}")


#Eq.23 (Modelo Global)
def calcular_D(p_star, P_c, rho):
    D_p_star = P_c + (1 / rho) * sum(p_star)
    return D_p_star

D_p_star = calcular_D(p_star, P_c, rho)
print(f"--D_p_star: {D_p_star}")

def calcular_lambda_estrela(tilde_C_p_star, D_p_star):
    # Calcula lambda* como a razão entre a capacidade de transmissão total e a potência total consumida
    lambda_estrela = tilde_C_p_star / D_p_star
    return lambda_estrela

lambda_estrela = calcular_D(p_star, P_c, rho)
print(f"--Eficiência energética máxima alcançável no sistema: {lambda_estrela}")


# Eq. 29 (Modelo Global)
def objetivo(p, W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, tilde_C_p_star, D_p_star, lambda_estrela):
    return -(tilde_C_p_star - lambda_estrela * D_p_star)

def resolver_problema_otimizacao(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, lambda_estrela):
    
    # Chute inicial: igualmente distribuído entre os feixes ativos
    initial_guess = [P_T / len(g_ru)] * len(g_ru)
    
    def constraint_total_power(p):
        # Restrição: a soma das potências dos feixes ativos deve ser menor ou igual à potência total disponível
        return sum(p) - P_T

    def constraint_individual_power(p):
        # Restrição: a potência individual de cada feixe ativo deve ser menor ou igual à potência individual máxima permitida
        return p - P_f

    def constraint_received_power(p):
        # Restrição: a soma das potências dos feixes ativos deve ser maior ou igual à potência recebida mínima
        return sum(p) - P_r / (g_s * g_b * L_b)

    # Definindo as restrições do problema de otimização
    constraints = [{'type': 'ineq', 'fun': constraint_total_power},
                   {'type': 'ineq', 'fun': constraint_individual_power},
                   {'type': 'ineq', 'fun': constraint_received_power}]
    
    # Chamada para o otimizador
    result = minimize(objetivo, initial_guess, args=(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, lambda_estrela, tilde_C_p_star, D_p_star),
                      constraints=constraints)
    
    return result

resultado_max = resolver_problema_otimizacao(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, lambda_estrela)
print(f"--Resultado da maximização: {resultado_max}")



#Algoritmo 3
# Inicialização
def initialization():
    """
    Inicializa o vetor de potências com valores aleatórios distribuídos entre os feixes ativos.
    """
    p_0 = np.random.uniform(0, P_T / len(g_ru), len(g_ru))
    return p_0

# Função objetivo para Dinkelbach
def objetivo_dinkelbach(p, W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, lambda_n):
    """
    Calcula a função objetivo do algoritmo de Dinkelbach.
    """
    tilde_C_p = tilde_C(p_star, g_t, g_ru, L, I_i, I_d, N_0, W, p_0)
    D_p = calcular_D(p_star, P_c, rho)
    return -(tilde_C_p - lambda_n * D_p)

# Otimização usando minimize
def resolver_problema_otimizacao_dinkelbach(lambda_n, p_0):
    """
    Resolve o problema de otimização usando a função objetivo de Dinkelbach e as restrições definidas.
    """
    constraints = [
        {'type': 'ineq', 'fun': lambda p: P_T - np.sum(p)},  # Soma das potências dos feixes ativos <= potência total disponível
        {'type': 'ineq', 'fun': lambda p: P_f - np.max(p)},  # Potência individual de cada feixe ativo <= potência individual máxima permitida
        {'type': 'ineq', 'fun': lambda p: P_r - np.sum(p) * g_s * g_b * L_b}  # Soma das potências dos feixes ativos >= potência recebida mínima
    ]

    result = minimize(
        objetivo_dinkelbach, 
        p_0, 
        args=(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, lambda_n), 
        constraints=constraints,
        method='SLSQP'  # Usando um método específico de otimização
    )

    return result

# Algoritmo de Dinkelbach
def dinkelbach_algorithm(p_0, epsilon=1e-5):
    """
    Executa o algoritmo de Dinkelbach para encontrar a solução ótima.
    """
    lambda_n = 0  # Inicializa o parâmetro lambda
    n = 0  # Contador de iterações

    while True:
        # Resolve o problema de otimização com o valor atual de lambda_n
        result = resolver_problema_otimizacao_dinkelbach(lambda_n, p_0)
        p_star = result.x  # Potências ótimas encontradas

        # Calcula tilde_C e D para as potências ótimas
        tilde_C_p_star = tilde_C(p_star, g_t, g_ru, L, I_i, I_d, N_0, W, p_0)
        D_p_star = calcular_D(p_star, P_c, rho)

        # Calcula F(lambda_n)
        F_lambda_n = tilde_C_p_star - lambda_n * D_p_star

        # Verifica a condição de parada
        if F_lambda_n < epsilon:
            break

        # Atualiza lambda_n e p_0 para a próxima iteração
        lambda_n = tilde_C_p_star / D_p_star
        p_0 = p_star
        n += 1

    return p_star, lambda_n

# Inicialização
p_0 = initialization()

# Executando o algoritmo de Dinkelbach
p_star, lambda_star = dinkelbach_algorithm(p_0)

print(f"--Potências ótimas dos feixes: {p_star}")
print(f"--Eficiência energética máxima alcançável no sistema: {lambda_star}")