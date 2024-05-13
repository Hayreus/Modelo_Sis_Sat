import math
import numpy as np
from scipy.optimize import minimize
from munkres import Munkres, print_matrix

# Parâmetros
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
def optimize_beam_power(epsilon, L, P_eq):
    # Parâmetros iniciais
    num_iterations = len(L)  # Determina o número máximo de iterações com base no tamanho de L
    p = [None] * num_iterations
    x = [None] * num_iterations
    
    # Configuração inicial
    i = 0
    p[i] = P_eq
    
    while i < num_iterations - 1:  # Usar 'num_iterations - 1' para garantir que não ultrapasse o índice máximo
        # Atualiza os valores de p e x
        x[i] = Algorithm2(p[i])  # Encontra o beam assignment com p fixo
        p[i + 1] = Algorithm3(x[i])    # Encontra a alocação de potência com x fixo
        
        # Verifica o critério de convergência
        if abs(eta(x[i], p[i]) - eta(x[i - 1], p[i - 1])) >= epsilon:  # Usar 'i - 1' para comparar com a iteração anterior
            break
    
        # Atualiza o índice para o próximo loop
        i += 1
    
    # Retorna as matrizes p* e x*
    return p[i], x[i]

# Algoritmo 2
def Algorithm2(p):
    m = Munkres()
    indexes = m.compute(p)
    x_star = indexes  # Atribuir os índices como a solução ótima
    return x_star


# Equações do algoritmo 2
# Eq. 12
def calcular_p_km(P_f, P_T, P_r, g_s, g_b, L_b, M):
    # Calcula p_km usando a fórmula fornecida
    Peq = min(P_f, P_T / M, P_r / (g_s * g_b * L_b * M))
    
    return P_eq

# Eq. 13
def constraint3(vars, P_s, P_b, L_b, P_r):
    X = vars[:K*M].reshape((K, M))
    p = vars[K*M:]
    
    # Calcula o lado esquerdo da desigualdade
    left_side = sum(sum(X[k][m] * p[k*M + m] for m in range(M)) for k in range(K))
    
    # Calcula o lado direito da desigualdade
    right_side = P_r / (g_s * g_b * L_b)
    
    # Retorna a desigualdade
    return left_side - right_side

# Eq. 14
def constraint7(vars):
    X = vars[:K*M].reshape((K, M))
    p = vars[K*M:]
    
    # Calcula o lado esquerdo da igualdade
    left_side = sum(sum(X[k][m] * p[k*M + m] for m in range(M)) for k in range(K))
    
    # Calcula o lado direito da igualdade
    right_side = sum(p)
    
    # Retorna a diferença entre os dois lados da igualdade
    return left_side - right_side

# Eq. 15
def calcular_eta(X, p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0):
    numerator = sum(sum(W * math.log2(1 + (p[m] * g_t * g_ru * L[k]) / (I_i[k][m] + I_d[k][m] + N_0 * W)) * X[k][m] for m in range(M)) for k in range(K))
    denominator = P_c + (1 / rho) * sum(p)
    
    eta = numerator / denominator
    
    return eta

# Eq. 16
def calcular_I_i(X, p, g_s, g_ru, L):
    I_i = [[0 for _ in range(M)] for _ in range(K)]
    
    for k in range(K):
        for m in range(M):
            # Calcula a interferência interna para cada par (k, m)
            I_i[k][m] = g_s * g_ru * L[k] * sum(p[m_prime] for m_prime in range(M) if m_prime != m)
    
    return I_i

# Eq. 17
def calcular_q_km(X, p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0):
    q = [[0 for _ in range(M)] for _ in range(K)]
    
    for k in range(K):
        for m in range(M):
            # Calcula o numerador do termo q_km
            numerator = W * math.log2(1 + (p[m] * gt * g_ru * L[k]) / (I_i[k][m] + I_d[k][m] + N_0 * W))
            
            # Calcula o denominador do termo q_km
            denominator = Pc + (1 / rho) * sum(p)
            
            # Calcula o termo q_km
            q[k][m] = numerator / denominator
    
    return q

# Eq. 18
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



# Algoritmo 3
def Algorithm3(x):
    p_0 = None  # Inicializar p_0
    while True:
        epsilon = 0  # Critério de parada
        n = 0
        lambda_n = 0
        while True:  #Dinkelbach's algorithm
            p_star = None 
            F_lambda_n = C_tilde(p_star) - lambda_n * D(p_star)
            lambda_n_plus_1 = C_tilde(p_star) / D(p_star)
            n += 1
            if F_lambda_n < epsilon:
                break
        if np.linalg.norm(np.array(p_0) - np.array(p_star)) < epsilon:
            break
        p_0 = p_star
    return p_0


# Funções das Equações para algoritmo 3
# Eq. 19
def calcular_eta(p, P_c, rho, W, g_t, g_ru, L, I_i, I_d, N_0):
    # Calcula o numerador da eficiência energética
    C_p = sum(W * math.log2(1 + (p[k] * g_t * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    
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

# Eq. 22
def objective_function(p, *args):
    W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho = args
    
    # Calcula a soma ponderada das taxas de transmissão de todos os feixes ativos
    numerator = sum(W * math.log2(1 + (p[k] * g_t * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    
    # Calcula o denominador da eficiência energética
    denominator = P_c + (1 / rho) * sum(p)
    
    # Calcula a eficiência energética como a razão entre o numerador e o denominador
    eta = numerator / denominator
    
    # Otimização do problema: maximizar a eficiência energética (equivalente a minimizar o negativo da função objetivo)
    return -eta

def constraint1(p, P_T):
    # Restrição: a soma das potências dos feixes ativos não pode exceder a potência total de transmissão PT
    return sum(p) - P_T

def constraint2(p, P_f):
    # Restrição: cada potência do feixe não pode exceder a potência máxima permitida Pf
    return [P_f - p_k for p_k in p]

def constraint3(p, P_r, g_s, g_b, L_b):
    # Restrição: a soma das potências dos feixes ativos não pode exceder a potência máxima permitida pelo receptor
    return P_r / (g_s * g_b * L_b) - sum(p)

def resolver_problema_otimizacao(W, g_t, g_ru, L, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b):
    # Número de feixes ativos
    num_feixes = len(g_t)
    
    # Chute inicial: igualmente distribuído entre os feixes ativos
    initial_guess = [P_T / num_feixes] * num_feixes
    
    # Definindo as restrições do problema de otimização
    constraints = [{'type': 'ineq', 'fun': constraint1, 'args': (P_T,)},
                   {'type': 'ineq', 'fun': constraint2, 'args': (P_f,)},
                   {'type': 'ineq', 'fun': constraint3, 'args': (P_r, g_s, g_b, L_b)}]
    
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
def calcular_R_p_til(k, p, p_0, f_1, f_2, grad_f2_p0):
    # Calcula f2(p0)
    f2_p0 = f_2(p_0)
    
    # Calcula o gradiente de f2(p0)
    grad_f2_p0_T = grad_f2_p0(p_0)
    
    # Calcula o termo dentro dos parênteses de f2(p_0) - grad(f2(p_0))^T * (p - p_0)
    termo_dentro_parenteses = f2_p0 - np.dot(grad_f2_p0_T, p - p_0)
    
    # Calcula f1(p) - (f2(p0) - grad(f2(p0))^T * (p - p0))
    tilde_R_k_p = f1(p) - termo_dentro_parenteses
    
    return tilde_R_k_p

# Eq.26
def calcular_C_p(p, f_1, f_2, W):
    # Calcula a soma ponderada das diferenças entre f1(p) e f2(p) para todos os feixes ativos
    C_p = sum(W * (f_1(p_k) - f_2(p_k)) for p_k in p)
    return C_p

# Eq.27
def calcular_f1(p, g_t, g_ru, L, I_i, I_d, N_0, W):
    # Calcula o valor de f1(p) para cada feixe
    f1_p = [np.log2(p_k * g_t * g_ru_k * L_k + I_i_k + I_d_k + N_0 * W) for p_k, g_ru_k, L_k, I_i_k, I_d_k in zip(p, g_ru, L, I_i, I_d)]
    return f1_p

# Eq.28
def calcular_f2(I_i, I_d, N_0, W):
    # Calcula o valor de f2(p) para cada feixe
    f2_p = [np.log2(I_i_k + I_d_k + N_0 * W) for I_i_k, I_d_k in zip(I_i, I_d)]
    return f2_p

# Eq.29
def objective_function(p, C_tilde_p, D_p, lambda_star):
    return -(C_tilde_p - lambda_star * D_p)

def constraint_total_power(p, P_T):
    return sum(p) - P_T

def constraint_individual_power(p, P_f):
    return p - P_f

def constraint_received_power(p, P_r, g_s, g_b, L_b):
    return sum(p) - P_r / (g_s * g_b * L_b)

def resolver_problema_otimizacao(C_tilde_p, D_p, lambda_star, P_T, P_f, P_r, g_s, g_b, L_b):
    initial_guess = [0.5] * len(C_tilde_p)
    bounds = [(0, Pf)] * len(C_tilde_p)

    constraints = [{'type': 'ineq', 'fun': constraint_total_power, 'args': (P_T,)},
                   {'type': 'ineq', 'fun': constraint_individual_power, 'args': (P_f,)},
                   {'type': 'ineq', 'fun': constraint_received_power, 'args': (P_r, g_s, g_b, L_b)}]

    result = minimize(objective_function, initial_guess, args=(C_tilde_p, D_p, lambda_star),
                      bounds=bounds, constraints=constraints)

    return result


## organizar equações do algoritmo 1