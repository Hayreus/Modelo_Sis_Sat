import math
import numpy as np
from scipy.optimize import minimize
from munkres import Munkres, print_matrix

# Parâmetros
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7.46                    # Velocidade orbital em km/s
F_c = 20                    # Frequencia de centro em Ghz
W = 28                      # Largura de banda em MHz
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
rho = 0.8                   # Eficiência do amplificador 


# Funções e equações dos algoritmos 1, 2 e 3
# Algoritmo 1
def Algorithm_1(epsilon, L, P_eq):
    # Parâmetros iniciais
    num_iterations = len(L)  # Determina o número máximo de iterações com base no tamanho de L
    p = [None] * num_iterations
    x = [None] * num_iterations
    
    # Configuração inicial
    i = 0
    p[i] = P_eq
    
    while i < num_iterations - 1:  # Usar 'num_iterations - 1' para garantir que não ultrapasse o índice máximo
        # Atualiza os valores de p e x
        x[i] = Algorithm_2(p[i])  # Encontra o beam assignment com p fixo
        p[i + 1] = Algorithm_3(x[i])    # Encontra a alocação de potência com x fixo
        
        # Verifica o critério de convergência
        if abs(eta(x[i], p[i]) - eta(x[i - 1], p[i - 1])) >= epsilon:  # Usar 'i - 1' para comparar com a iteração anterior
            break
    
        # Atualiza o índice para o próximo loop
        i += 1
    
    # Retorna as matrizes p* e x*
    return p[i], x[i]

# Algoritmo 2
def Algorithm_2(p):
    m = Munkres()
    indexes = m.compute(p)
    x_star = indexes  # Atribuir os índices como a solução ótima
    return x_star


# Funções das Equações para algoritmo 2
# Eq. 12
def calcular_p_km(P_f, P_T, P_r, g_s, g_b, L_b, M):
    p_km = {}

    # Para cada combinação de feixe k e ponto de acesso m
    for k in range(1, K + 1):
        for m in range(1, M + 1):
            # Calcula p_{k,m} como o mínimo entre três valores
            p_km[(k, m)] = min(P_f, P_T / M, P_r / (g_s * g_b * L_b * M))
    
    return p_km

# Eq. 13
def constraint3(vars, P_s, P_b, L_b, P_r, g_s, g_b):
    K, M = len(P_s), len(P_b) 
    X = np.array(vars[:K * M]).reshape((K, M))
    p = np.array(vars[K * M:])

    # Calcula o lado esquerdo da desigualdade
    left_side = sum(sum(X[k][m] * p[k * M + m] for m in range(M)) for k in range(K))
    
    # Calcula o lado direito da desigualdade
    right_side = P_r / (g_s * g_b * L_b)
    
    # Retorna a desigualdade
    return left_side - right_side

# Eq. 14
def constraint7(vars, K, M):
    # Extrair X e p do vetor vars
    X = np.array(vars[:K * M]).reshape((K, M))
    p = np.array(vars[K * M:K * M + M])
    
    # Calcula o lado esquerdo da igualdade
    left_side = sum(sum(X[k, m] * p[m] for m in range(M)) for k in range(K))
    
    # Calcula o lado direito da igualdade
    right_side = sum(p)
    
    # Retorna a diferença entre os dois lados da igualdade
    return left_side - right_side

# Eq. 15
def objective_function(p, W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho):
    numerator = sum(W * math.log2(1 + (p[k] * g_t * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    denominator = P_c + (1 / rho) * sum(p)
    eta = numerator / denominator
    return -eta

# Eq. 16
def calcular_I_i(X, p, g_s, g_ru, L):
    K, M = len(X), len(p)
    I_i = [0] * M

    for m in range(M):
        for m_prime in range(M):
            if m_prime != m:
                I_i[m] += p[m_prime]
        I_i[m] *= g_s * g_ru * L[m]
    
    return I_i

# Eq. 17
def calcular_q(X, p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0):

    K, M = X.shape
    q = np.zeros((K, M))
    
    # Calcula o denominador da equação
    denominator = P_c + (1 / rho) * np.sum(p)
    
    for k in range(K):
        for m in range(M):
            # Calcula o numerador da equação
            numerator = p[m] * g_t * g_ru[k] * L[k]
            
            # Calcula o termo de interferência total
            interference = I_i[k][m] + I_d[k][m] + N_0 * W
            
            # Calcula q_{k,m}
            q[k][m] = (W / denominator) * np.log2(1 + numerator / interference)
    
    return q

# Eq. 18 (restrições)
def objective_function(X, q):
    
    return -sum(sum(X[k][m] * q[k][m] for m in range(M)) for k in range(K))

def constraint1(X):
    
    return [sum(X[k]) - 1 for k in range(K)]

def constraint2(X):
    return [sum(X[:, m]) - 1 for m in range(M)]

def constraint3(X):
    return sum(sum(X)) - M

def resolver_problema_otimizacao(q, K, M):
    initial_guess = [[0.5] * M for _ in range(K)]
    bounds = [(0, 1) for _ in range(K * M)]

    constraints = [{'type': 'eq', 'fun': constraint1},
                   {'type': 'eq', 'fun': constraint2},
                   {'type': 'eq', 'fun': constraint3}]

    result = minimize(objective_function, initial_guess, args=(q,),
                      bounds=bounds, constraints=constraints)
    return result


# Funções das Equações para algoritmo 3
# Eq. 19
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

# Eq. 20
def calcular_I_i(p, g_s, g_ru, L):
    I_i = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula a interferência interna para o feixe k
        I_i[k] = g_s * g_ru[k] * L[k] * sum(p[k_prime] for k_prime in range(len(p)) if k_prime != k)
    
    return I_i

# Eq. 21
def calcular_I_d(p, g_t, g_ru, L, f_k, T_s):
    I_d = [0] * len(p)

    for k in range(len(p)):
        # Calcula o termo sinc(f_k * T_s)
        sinc_term = math.sin(math.pi * f_k[k] * T_s) / (math.pi * f_k[k] * T_s) if f_k[k] * T_s != 0 else 1
        
        # Calcula a interferência externa para o feixe k
        I_d[k] = p[k] * g_t * g_ru[k] * L[k] * (1 - sinc_term**2)
    
    return I_d

# Eq. 22 (restrições)
def objective_function(p, W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho):
    numerator = sum(W * math.log2(1 + (p[k] * g_t * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    denominator = P_c + (1 / rho) * sum(p)
    eta = numerator / denominator
    return -eta

def objective_function(p, W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho):
    numerator = sum(W * math.log2(1 + (p[k] * g_t * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    denominator = P_c + (1 / rho) * sum(p)
    eta = numerator / denominator
    return -eta

def resolver_problema_otimizacao(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b):
    # Número de feixes ativos
    num_feixes = len(g_t)
    
    # Chute inicial: igualmente distribuído entre os feixes ativos
    initial_guess = [P_T / num_feixes] * num_feixes
    
    #definindo as restrições do problema de otimização
    def constraint_total_power(p):
        return sum(p) - P_T

    def constraint_individual_power(p):
        return p - P_f

    def constraint_received_power(p):
        return sum(p) - P_r / (g_s * g_b * L_b)
                
    # Lista de restrições
    constraints = [{'type': 'ineq', 'fun': constraint_total_power},
                   {'type': 'ineq', 'fun': constraint_individual_power},
                   {'type': 'ineq', 'fun': constraint_received_power}]
    
    # Chamada para o otimizador
    result = minimize(objective_function, initial_guess, args=(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho),
                      constraints=constraints)
    
    return result

# Eq. 23
def calcular_lambda_estrela(C_p_estrela, D_p_estrela):
    # Calcula lambda* como a razão entre a capacidade de transmissão total e a potência total consumida
    lambda_estrela = C_p_estrela / D_p_estrela
    return lambda_estrela

# Eq.24
def calcular_C_p_estrela(W, R_p_estrela):
    # Calcula a capacidade de transmissão total no ponto ótimo p* como a soma ponderada das taxas de transmissão de todos os feixes ativos
    C_p_estrela = W * sum(R_p_estrela)
    return C_p_estrela

# Eq.25
def grad_f2(I_i, I_d, N_0, W):

    K = len(I_i)  # Número de feixes
    grad_values = np.zeros(K)
    
    for k in range(K):
        interference = I_i[k] + I_d[k] + N_0 * W
        grad_values[k] = 1 / (np.log(2) * interference)  # Derivada de log2(interference)
    
    return grad_values


def calculate_tilde_R(p, p_0, g_t, g_ru, L, I_i, I_d, N_0, W):
 
    K = len(p)  # Número de feixes
    tilde_R = np.zeros(K)
    
    f1_values = f1(p, g_t, g_ru, L, I_i, I_d, N_0, W)
    f2_values = f2(I_i, I_d, N_0, W)
    grad_f2_values = grad_f2(I_i, I_d, N_0, W)
    
    for k in range(K):
        gradient_term = grad_f2_values[k] * (p[k] - p_0[k])
        tilde_R[k] = f1_values[k] - (f2_values[k] - gradient_term)
    
    return tilde_R

# Eq.26
def calcular_C_p(p, f_1, f_2, W):

    C_p = 0
    for k in range(len(p)):
        p_k = p[k]
        C_p += W * (f_1(p_k) - f_2(p_k))
    return C_p

# Eq.27
def f1(p, g_t, g_ru, L, I_i, I_d, N_0, W):

    f1_values = np.zeros(len(p))
    
    for k in range(len(p)):
        numerator = p[k] * g_t * g_ru[k] * L[k]
        denominator = I_i[k] + I_d[k] + N_0 * W
        f1_values[k] = np.log2(1 + numerator / denominator)
    
    return f1_values

# Eq.28
def f2(I_i, I_d, N_0, W):

    f2_values = np.zeros(len(I_i))
    
    for k in range(len(I_i)):
        interference = I_i[k] + I_d[k] + N_0 * W
        f2_values[k] = np.log2(interference)
    
    return f2_values

# Eq.29 (restrições)
def resolver_problema_otimizacao(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b):
    # Número de feixes ativos
    num_feixes = len(g_t)
    
    # Chute inicial: igualmente distribuído entre os feixes ativos
    initial_guess = [P_T / num_feixes] * num_feixes
    
    def constraint_total_power(p):
        return sum(p) - P_T

    def constraint_individual_power(p):
        return p - P_f

    def constraint_received_power(p):
        return sum(p) - P_r / (g_s * g_b * L_b)

    # Definindo as restrições do problema de otimização
    constraints = [{'type': 'ineq', 'fun': constraint_total_power, 'args': (P_T,)},
                   {'type': 'ineq', 'fun': constraint_individual_power, 'args': (P_f,)},
                   {'type': 'ineq', 'fun': constraint_received_power, 'args': (P_r, g_s, g_b, L_b)}]
    
    # Chamada para o otimizador
    result = minimize(objective_function, initial_guess, args=(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho),
                      constraints=constraints)
    
    return result


# Algoritmo 3

def C_tilde(p, g_t, g_ru, L, I_i, I_d, N_0, W):
    K = len(p)
    c_tilde = 0
    for k in range(K):
        numerator = p[k] * g_t * g_ru[k] * L[k]
        denominator = I_i[k] + I_d[k] + N_0 * W
        c_tilde += W * np.log2(1 + numerator / denominator)
    return c_tilde

def D(p, P_c, rho):
    return P_c + (1 / rho) * sum(p)

def Algorithm_3(x, W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b):
    epsilon = 1e-6  # Critério de parada
    p_0 = np.zeros(len(x))  # Inicializar p_0
    while True:
        n = 0
        lambda_n = 0
        while True:  # Dinkelbach's algorithm
            result = resolver_problema_otimizacao(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b)
            p_star = result.x
            F_lambda_n = C_tilde(p_star, g_t, g_ru, L, I_i, I_d, N_0, W) - lambda_n * D(p_star, P_c, rho)
            lambda_n_plus_1 = C_tilde(p_star, g_t, g_ru, L, I_i, I_d, N_0, W) / D(p_star, P_c, rho)
            n += 1
            if F_lambda_n < epsilon:
                break
            lambda_n = lambda_n_plus_1
        if np.linalg.norm(np.array(p_0) - np.array(p_star)) < epsilon:
            break
        p_0 = p_star
    return p_0