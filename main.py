import math
import numpy as np
from scipy.optimize import minimize
from munkres import Munkres, print_matrix

# Algoritmo 1
def Algorithm1(epsilon, L, Peq):
    i = 0
    p_prev = Peq
    while True:
        p_prev = Peq
        x_i = Algorithm2(p_prev)
        p_i = Algorithm3(x_i)
        p_star = p_i
        x_star = x_i
        if abs(eta(x_i, p_i) - eta(x_prev, p_prev)) >= epsilon:
            break
    return p_star, x_star

# Algoritmo 2
def Algorithm2(p):
    m = Munkres()
    indexes = m.compute(p)
    x_star = indexes  # Atribuir os índices como a solução ótima
    return x_star


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

# Eq. 19
def calcular_eta(p, Pc, rho, W, gt, g_ru, L, I_i, I_d, N_0):
    # Calcula o numerador da eficiência energética
    C_p = sum(W * math.log2(1 + (p[k] * gt * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    
    # Calcula o denominador da eficiência energética
    D_p = Pc + (1 / rho) * sum(p)
    
    # Calcula a eficiência energética como a razão entre o numerador e o denominador
    eta = C_p / D_p
    
    return eta

# Eq. 20
def calcular_I_i(p, gs, g_ru, L):
    I_i = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula a interferência interna para o feixe k
        I_i[k] = gs * g_ru[k] * L[k] * sum(p[k_prime] for k_prime in range(len(p)) if k_prime != k)
    
    return I_i

# Eq. 21
def calcular_I_d(p, gt, g_ru, L, f_k, T_s):
    I_d = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula o termo sinc(f_k * T_s)
        sinc_term = math.sin(math.pi * f_k[k] * T_s) / (math.pi * f_k[k] * T_s) if f_k[k] * T_s != 0 else 1
        
        # Calcula a interferência externa para o feixe k
        I_d[k] = p[k] * gt * g_ru[k] * L[k] * (1 - sinc_term**2)
    
    return I_d

# Eq. 22
def objective_function(p, *args):
    W, gt, g_ru, L, I_i, I_d, N_0, Pc, rho = args
    
    # Calcula a soma ponderada das taxas de transmissão de todos os feixes ativos
    numerator = sum(W * math.log2(1 + (p[k] * gt * g_ru[k] * L[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(len(p)))
    
    # Calcula o denominador da eficiência energética
    denominator = Pc + (1 / rho) * sum(p)
    
    # Calcula a eficiência energética como a razão entre o numerador e o denominador
    eta = numerator / denominator
    
    # Otimização do problema: maximizar a eficiência energética (equivalente a minimizar o negativo da função objetivo)
    return -eta

def constraint1(p, PT):
    # Restrição: a soma das potências dos feixes ativos não pode exceder a potência total de transmissão PT
    return sum(p) - PT

def constraint2(p, Pf):
    # Restrição: cada potência do feixe não pode exceder a potência máxima permitida Pf
    return [Pf - pk for pk in p]

def constraint3(p, Pr, gs, gb, Lb):
    # Restrição: a soma das potências dos feixes ativos não pode exceder a potência máxima permitida pelo receptor
    return Pr / (gs * gb * Lb) - sum(p)

def resolver_problema_otimizacao(W, gt, g_ru, L, I_i, I_d, N_0, Pc, rho, PT, Pf, Pr, gs, gb, Lb):
    # Número de feixes ativos
    num_feixes = len(gt)
    
    # Chute inicial: igualmente distribuído entre os feixes ativos
    initial_guess = [PT / num_feixes] * num_feixes
    
    # Definindo as restrições do problema de otimização
    constraints = [{'type': 'ineq', 'fun': constraint1, 'args': (PT,)},
                   {'type': 'ineq', 'fun': constraint2, 'args': (Pf,)},
                   {'type': 'ineq', 'fun': constraint3, 'args': (Pr, gs, gb, Lb)}]
    
    # Chamada para o otimizador
    result = minimize(objective_function, initial_guess, args=(W, gt, g_ru, L, I_i, I_d, N_0, Pc, rho),
                      constraints=constraints)
    
    return result

# Eq. 23
def calcular_lambda_estrela(C_p_estrela, D_p_estrela):
    # Calcula lambda* como a razão entre a capacidade de transmissão total e a potência total consumida
    lambda_estrela = C_p_estrela / D_p_estrela
    return lambda_estrela