import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment
from munkres import Munkres, print_matrix

np.set_printoptions(threshold=np.inf)

# Parâmetros
N = 1                       # Número de niveis de camada hexagonais
n =  7                      # Número de celulas hexagonais
num_usuario_por_celula = 2
usuarios = num_usuario_por_celula * n
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7460                    # Velocidade orbital em m/s
F_c = 20e9                  # Frequencia de centro em hz
W = 28e6                    # Largura de banda em MHz 28 (28e6 Hz)
T_s = 1e-6                  # Tempo de duração do símbolo em micro segundo
micro = -2.6                # Parâmetro de desvanecimento da chuva em dB*
sigma =  1.6                # Parâmetro de desvanecimento da chuva em dB*
N_0 = 10**(-172/10)         # Densidade espetral do ruído em dBw/Hz para W/Hz
M = usuarios                # Número de feixes
g_t = 10**(52.1/10)         # Ganho da antena do satélite em W
g_s = 10**(5/10)            # Lóbulo lateral da antena de satélite em dB
g_ru = [10**(10/10),
        10**(11/10),
        10**(12/10),
        10**(13/10),
        10**(14/10),
        10**(15/10)]        # Ganho da antena dos usuários, intervalo de 10 a 15 com passo de 1 em dB

g_b = 10**(5/10)            # Ganho da estação de base em w
P_f = 10**(10/10)           # Potência máxima transmitida em w
P_r = np.full(M, 10**(-111/10))         # Potência de interferência admissível em w
P_c = 10**(10/10)           # Dissipação de potência do circuito em dBw
rho = 0.8                   # Eficiência do amplificador 
R = 6371                    # Raio médio da Terra em Km
xi = 15 * math.pi / 180     # Angulo minimo de elevação dado em graus e convertido para radianos
L_b = 1                     # Perda de transmissão

# Dados de teste

P_T = 10**(30/10)                   # Potência total de transmissão em W


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
print (f"--Ângulo de cobertura em radianos: {psi:.4f}")


#Eq. 2 (Posição Global)
def calcular_beta(psi, n):
    beta = (2 * psi) / (2 * n + 1)
    
    return beta


beta = calcular_beta(psi, n)
print(f"--Angulo de cobertura de cada célula: {beta:.4f}")


#Eq. 3 (Posição Global)
def calcular_Nc(N):
    Nc = 1 + (6 * N * (N + 1)) / 2
    
    return Nc


Nc = calcular_Nc(N)
print(f"--Número de feixes pontuais: {Nc}")


#Eq. 4 (Posição Global)
def calcular_theta_0(R, h, beta):
    numerador = R * math.sin(beta / 2)
    denominador = h + R - R * math.cos(beta / 2)
    theta_0 = (math.atan2(numerador, denominador))
    
    return theta_0


theta_0 = calcular_theta_0(R, h, beta)
print(f"--largura do feixe da célula central: {theta_0:.4f}")


#Eq. 5 (Posição Global)
def calcular_theta_n(R, h, beta, theta_0, N):

    thetas = np.zeros(N + 1)
    for n in range(N + 1):
        if n == 0:
            thetas[n] = theta_0
        else:
            beta_term = (2 * n + 1) * beta / 2
            numerator = R * np.sin(beta_term)
            denominator = h + R - R * np.cos(beta_term)
            atan_term = np.arctan2(numerator, denominator)
            
            # Soma das larguras dos feixes anteriores até n-1
            theta_k_sum = np.sum(thetas[:n])
            theta_n = atan_term - theta_k_sum - theta_0 / 2
            
            thetas[n] = theta_n
            
    
    return thetas


theta_n1 = calcular_theta_n(R, h, beta, theta_0, N)
theta_n = [theta_n1[0]/2, theta_n1[1]/2, theta_n1[1]/2, theta_n1[1]/2, theta_n1[1]/2, theta_n1[1]/2, theta_n1[1]/2]

print(f"--Abertura θ das células em radianos:\n{theta_n}")


def calcular_area_calota_esferica(R, psi):

    area_calota_esferica = 2 * math.pi * R ** 2 * (1 - math.cos(psi))
    return area_calota_esferica


# Calcular a área da calota esférica
area_calota_esferica = calcular_area_calota_esferica(R, psi)
print(f"--Área da calota esférica: {area_calota_esferica:.4f} km²")


def calcular_comprimento_arco(R, psi):

    L = R * psi
    return L


# Calcular o comprimento do arco do setor circular
raio = (calcular_comprimento_arco(R, psi))/6
Raio_Total = calcular_comprimento_arco(R, psi)/2
diametro = calcular_comprimento_arco(R, psi)
print(f"--Raio da área de cobertura da célula: {raio:.4f} km")
print(f"--Raio da área de cobertura Total: {Raio_Total:.4f} km")
print(f"--Diâmetro da área de cobertura Total: {diametro:.4f} km")


def calcular_pontos_hexagonais(raio):
    angulos = np.linspace(0, 2 * np.pi, 7)[:-1]  # Divide o círculo em 6 partes iguais
    pontos = [(2 * raio * np.cos(angulo), 2 * raio * np.sin(angulo)) for angulo in angulos]
    return pontos

def gerar_pontos_no_circulo(centro, raio, num_pontos):
    pontos = []
    for _ in range(num_pontos):
        r = raio * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        x = centro[0] + r * np.cos(theta)
        y = centro[1] + r * np.sin(theta)
        pontos.append((x, y))
    return pontos

def plotar_circulos_e_pontos(raio, num_pontos_por_circulo):
    fig, ax = plt.subplots()
    
    # Lista para armazenar as coordenadas de todos os pontos
    todas_coordenadas = []
    
    # Adiciona o círculo central e seus pontos
    circulo_central = plt.Circle((0, 0), raio * 1.15, edgecolor='r', facecolor='none', linestyle='--', label='Cobertura dos Feixes')
    ax.add_artist(circulo_central)
    
    pontos_central = gerar_pontos_no_circulo((0, 0), raio, num_pontos_por_circulo)
    todas_coordenadas.extend(pontos_central)
    for ponto in pontos_central:
        ax.plot(ponto[0], ponto[1], 'b.', alpha=0.5, label='Usuários' if todas_coordenadas.index(ponto) == 0 else "")
    
    # Adiciona os círculos ao redor e seus pontos
    pontos_hexagonais = calcular_pontos_hexagonais(raio)
    for centro in pontos_hexagonais:
        circulo = plt.Circle(centro, raio * 1.15, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_artist(circulo)
        
        pontos = gerar_pontos_no_circulo(centro, raio, num_pontos_por_circulo)
        todas_coordenadas.extend(pontos)
        for ponto in pontos:
            ax.plot(ponto[0], ponto[1], 'b.', alpha=0.5)
    
    # Adiciona o círculo externo
    Raio_Total = 2 * raio + raio
    circulo_externo = plt.Circle((0, 0), Raio_Total, edgecolor='g', facecolor='none', linestyle='-', label='Cobertura do Satélite')
    ax.add_artist(circulo_externo)
    
    # Configurações do gráfico
    ax.set_xlim(-4 * raio, 4 * raio)
    ax.set_ylim(-4 * raio, 4 * raio)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title("Distribuição de 7 Células com Usuários ou Bases Aleatórias")
    plt.xlabel("Direção Horizontal X")
    plt.ylabel("Direção Vertical Y")
    
    # Adiciona a legenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()
    
    return todas_coordenadas


# Gerar e plotar as coordenadas dos usuários
coordenadas_todos_usuarios = plotar_circulos_e_pontos(raio, num_usuario_por_celula)
print(f"--Coordenada de Usuários:\n{coordenadas_todos_usuarios}")
print(f"--Número Total de Usuários:\n{len(coordenadas_todos_usuarios)}")


def converter_xy_para_lat_long(coordenadas_todos_usuarios, R):
    
    # Lista para armazenar as latitudes e longitudes calculadas
    coordenadas_lat_long = []
    
    # Itera sobre cada ponto e calcula a latitude e longitude correspondentes
    for ponto in coordenadas_todos_usuarios:
        x, y = ponto
        # Conversão de coordenadas x e y para latitude e longitude em radianos
        latitude = np.arctan(np.sinh(y / R))
        longitude = x / R
        
        coordenadas_lat_long.append((latitude, longitude))
    
    return coordenadas_lat_long


coordenadas_lat_long = converter_xy_para_lat_long(coordenadas_todos_usuarios, R)
print(f"Coordenadas em latitude e longitude (radianos):\n{coordenadas_lat_long}")


def calcular_distancias_satelite_para_pontos(coordenadas_lat_long, h, R):
    
    # Lista para armazenar as distâncias para cada ponto
    distancias = []
    
    # Calcular a distância para cada ponto
    for phi, lambda_ in coordenadas_lat_long:
        # Coordenadas do ponto
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_lambda = np.cos(lambda_)
        sin_lambda = np.sin(lambda_)
        
        # Calcular a distância usando teorema de Pitágoras (que não é dele de verdade, era dos Egípcios)
        d2 = (R + h - R * cos_lambda * cos_phi) ** 2 + (R * sin_lambda * cos_phi) ** 2 + (R * sin_phi) ** 2
        distancia = np.sqrt(d2)
        distancias.append(distancia)
    
    return distancias


print(f"Distâncias: {calcular_distancias_satelite_para_pontos(coordenadas_lat_long, h, R)}")

############################################################################################




##### Equações para algoritmo 3

#Eq.52 (Modelo Global)
def calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M):

    lambda_ = c / F_c  # Comprimento de onda
    L_k = np.zeros(M)
    
    for k in range(M):
        selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_k da lista
        d_m = distancias[k] * 1000  # Converte a distância de km para m
        d_k = P_T * (g_t * selected_g_ru * lambda_**2) / ((4 * np.pi * d_m)**2 * L_b)
        h_k = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)  # Variável aleatória complexa
        L_k[k] = d_k * np.abs(h_k)**2
    
    return L_k




#Eq. 5 (Modelo Global)
def calcular_phi_k(coordenadas_lat_long, lat_sat=0, lon_sat=0):
    
    # Coordenadas do satélite em cartesianas
    x_sat = np.cos(lat_sat) * np.cos(lon_sat)
    y_sat = np.cos(lat_sat) * np.sin(lon_sat)
    z_sat = np.sin(lat_sat)
    
    # Lista para armazenar os valores de phi_k
    phi_k_list = []
    
    for lat_k, lon_k in coordenadas_lat_long:
        # Coordenadas do ponto k em cartesianas
        x_k = np.cos(lat_k) * np.cos(lon_k)
        y_k = np.cos(lat_k) * np.sin(lon_k)
        z_k = np.sin(lat_k)
        
        # Vetor do satélite para o ponto k
        delta_x = x_k - x_sat
        delta_y = y_k - y_sat
        delta_z = z_k - z_sat
        
        # Magnitude dos vetores
        mag_sat = np.sqrt(x_sat**2 + y_sat**2 + z_sat**2)
        mag_k = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        
        # Produto escalar
        dot_product = x_sat * delta_x + y_sat * delta_y + z_sat * delta_z
        
        # Calcular cos(phi_k)
        cos_phi_k = dot_product / (mag_sat * mag_k)
        
        # Calcular phi_k e adicionar à lista
        phi_k = np.arccos(cos_phi_k)
        phi_k_list.append(phi_k)
    
    return np.array(phi_k_list)


phi_k_list = calcular_phi_k(coordenadas_lat_long)
print(f"--Eq.5 phi_k para cada ponto em radianos:\n{phi_k_list}")




###### Função para calcular f_k
def calcular_fk(v, F_c, c, phi_k_list, M):

    f_k = []
    for k in range(M):
        angle = phi_k_list[k]
        f_k.append((v * F_c / c) * np.cos(angle))
    return f_k


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

def optimize_energy_efficiency(K, M, P_T, P_f, P_r, g_s, g_b, L_b):

    # Calcula p_e0
    p_e = calculate_pe(P_f, P_T, P_r, g_s, g_b, L_b, M)
    
    # Definindo a função objetivo (a ser maximizada, mas linprog minimiza, então invertemos o sinal)
    c = -np.ones(K * M)  # Maximizar η(X, p_e) onde η é proporcional ao número de alocações
    
    # Definindo as restrições
    A_eq = np.zeros((K + M + 1, K * M))
    b_eq = np.zeros(K + M + 1)
    
    # Restrição 32d: Cada usuário pode ser alocado a no máximo um feixe
    for k in range(K):
        for m in range(M):
            A_eq[k, k * M + m] = 1
        b_eq[k] = 1
    
    # Restrição 32e: Cada feixe pode atender no máximo um usuário de cada vez
    for m in range(M):
        for k in range(K):
            A_eq[K + m, k * M + m] = 1
        b_eq[K + m] = 1
    
    # Restrição 32f: Número total de alocações não pode exceder M
    for k in range(K):
        for m in range(M):
            A_eq[K + M, k * M + m] = 1
    b_eq[K + M] = M
    
    # Definindo as desigualdades
    A_ub = np.zeros((1, K * M))
    b_ub = np.zeros(1)
    
    # Restrição 32a: Potência total alocada deve ser menor ou igual a P_T
    for k in range(K):
        for m in range(M):
            A_ub[0, k * M + m] = p_e0
    b_ub[0] = P_T
    
    # Resolvendo o problema de otimização
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
    
    # Organizando o resultado
    if result.success:
        allocation_matrix = result.x.reshape((K, M))
        return {
            "success": result.success,
            "allocation_matrix": allocation_matrix,
            "p_e": p_e,
            "fun": -result.fun  # Inverter o sinal de volta, pois minimizamos -η
        }
    else:
        return {
            "success": result.success,
            "message": result.message
        }


#####Eq.21 (Modelo Global)
def calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M):

    I_d = np.zeros(M)

    for k in range(M):
        selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_ru
        I_d[k] = p_e[k] * g_t * selected_g_ru * L_k[k] * (1 - np.sinc(f_k[k] * T_s)**2)
    return I_d


#####Eq.20 (Modelo Global)
def calcular_I_i(p_e, g_s, g_ru, L_k, P_f, P_T, P_r, g_b, L_b, M):

    I_i = np.zeros(M)
    
    for k in range(M):
        selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_ru
        sum_p_k_prime = p_e[k] * np.sum([1 if k_prime != k else 0 for k_prime in range(M)])
        I_i[k] = g_s * selected_g_ru * L_k[k] * sum_p_k_prime
        
    return I_i


#Eq.19 (Modelo Global)
def calcular_eta(M, p_e, P_c, rho, W, g_t, g_ru, L_k, I_i, I_d, N_0):
    
    # Calcula o numerador da eficiência energética
    C_p = 0
    for k in range(M):
        selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_ru
        numerator = p_e[k] * g_t * selected_g_ru * L_k[k]
        denominator = I_i[k] + I_d[k] + N_0 * W
        C_p += W * math.log2(1 + numerator / denominator)
    
    # Calcula o denominador da eficiência energética
    D_p = P_c + (1 / rho) * p_e * M
    
    # Calcula a eficiência energética como a razão entre o numerador e o denominador
    eta = C_p / D_p
    
    return eta


#Eq. 22 (Modelo Global)

# Definição da função objetivo (Eq. 22 - Modelo Global)
def optimize_power_allocation(p_e, W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, M, P_T, P_f, P_r, g_s, g_b, L_b):
    def objective(p_e):
        selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_ru
        num = np.sum([W * np.log2(1 + (p_e[k] * g_t * selected_g_ru * L_k[k]) / (I_i[k] + I_d[k] + N_0 * W)) for k in range(M)])
        denom = P_c + (1 / rho) * np.sum(p_e)
        return -num / denom  # Negativo porque estamos maximizando

    # Restrições
    def constraint1(p_e):
        return P_T - np.sum(p_e)

    def constraint2(p_e):
        return P_f - np.max(p_e)

    def constraint3(p_e):
        return P_r - np.sum(p_e) * g_s * g_b * L_b

    # Inicialização das potências (valores iniciais)
    p_0 = np.ones(M) * (P_T / M)

    # Definição das restrições
    cons = [
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}
    ]

    # Limites para as potências
    bounds = [(0, P_f) for _ in range(M)]

    # Resolução do problema de otimização
    solution = minimize(objective, p_0, method='SLSQP', bounds=bounds, constraints=cons)

    return solution




######Eq.27 (Modelo Global)
def f1(p_e, g_t, g_ru, L_k, I_i, N_0, W, f_k, T_s, P_f, P_T, P_r, g_s, g_b, L_b, M):

    I_d = calculate_I_d(p_e, g_t, g_ru, T_s, f_k, L_k, P_f, P_T, P_r, g_s, g_b, L_b, M)
    f1_values = np.zeros(M)

    for k in range(M):
        selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_ru
        numerator = p_e[k] * g_t * selected_g_ru * L_k[k]
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


def calcular_lambda_estrela(tilde_C_p_star, D_p_star):
    # Calcula lambda* como a razão entre a capacidade de transmissão total e a potência total consumida
    lambda_estrela = tilde_C_p_star / D_p_star
    return lambda_estrela


#Eq.29 (Modelo Global)
def objetivo(p_e_1, W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, tilde_C_p_star, D_p_star, lambda_estrela):
    return -(tilde_C_p_star - lambda_estrela * D_p_star)

def resolver_problema_otimizacao(W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, lambda_estrela):
    
    # Chute inicial: igualmente distribuído entre os feixes ativos
    initial_guess = [P_T / len(g_ru)] * len(g_ru)
    
    def constraint_total_power(p_e_1):
        # Restrição: a soma das potências dos feixes ativos deve ser menor ou igual à potência total disponível
        return sum(p_e_1) - P_T
    
    def constraint_individual_power(p_e_1):
        # Restrição: a potência individual de cada feixe ativo deve ser menor ou igual à potência individual máxima permitida
        return p_e_1 - P_f

    def constraint_received_power(p_e_1):
        # Restrição: a soma das potências dos feixes ativos deve ser maior ou igual à potência recebida mínima
        return sum(p_e_1) - P_r / (g_s * g_b * L_b)

    # Definindo as restrições do problema de otimização
    constraints = [{'type': 'ineq', 'fun': constraint_total_power},
                   {'type': 'ineq', 'fun': constraint_individual_power},
                   {'type': 'ineq', 'fun': constraint_received_power}]
    
    # Chamada para o otimizador
    result = minimize(objetivo, initial_guess, args=(W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, P_T, P_f, P_r, g_s, g_b, L_b, lambda_estrela, tilde_C_p_star, D_p_star),
                      constraints=constraints)
    
    return result





########################################################################################################################################
# Algoritmo 3

# Função objetivo para Dinkelbach
def objetivo_dinkelbach(p_0, p_e, W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, lambda_n, M):
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
    
    # Define as restrições
    constraints = [
        {'type': 'ineq', 'fun': lambda p: P_T - np.sum(p)},  # Soma das potências dos feixes ativos <= potência total disponível
        {'type': 'ineq', 'fun': lambda p: P_f - np.max(p)},  # Potência individual de cada feixe ativo <= potência individual máxima permitida
        {'type': 'ineq', 'fun': lambda p: np.sum(p) * g_s * g_b * L_b - P_r}  # Soma das potências dos feixes ativos >= potência recebida mínima
    ]

    # Resolve o problema de otimização usando a função objetivo de Dinkelbach
    result = minimize(
        objetivo_dinkelbach,
        p_0,
        args=(p_e, W, g_t, g_ru, L_k, I_i, I_d, N_0, P_c, rho, lambda_n, M),
        constraints=constraints,
        method='SLSQP'
    )
    return result

# Algoritmo de Dinkelbach
def dinkelbach_algorithm(p_0, c, P_T, g_t, g_ru, F_c, distancias, L_b, P_f, P_r, g_s, g_b, M, T_s, v, phi_k_list, usuarios, P_c, rho, N_0, W, epsilon=1e-5):
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

    return p_star, lambda_n




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
            selected_g_ru = np.random.choice(g_ru)
            numerator += W * np.log2(1 + (P_r[m] * g_t * selected_g_ru * L_k[k]) / (I_i[k] + I_d[m] + N_0 * W)) * X[k, m]
    denominator = P_c + (1 / rho) * np.sum(P_r)
    eta = numerator / denominator
    return eta


# Eq.16 (Modelo Global), interferência entre feixes
def calcular_I_ki(g_s, g_ru, L_k, p_km, x_km, M):
    I_ki = np.zeros((M, M))
    for k in range(M):
        for m in range(M):
            selected_g_ru = np.random.choice(g_ru)  # Seleciona aleatoriamente um valor de g_ru
            interferencia = g_s * selected_g_ru* L_k[k] * np.sum(p_km[:, m] * (1 - x_km[:, m]))
            I_ki[k, m] = interferencia
    return I_ki


# Eq.17 (Modelo Global)
def calcular_q_km(W, P_c, rho, p_km, g_t, g_ru, L_k, I_ki, I_d, N_0):
    q_km = np.zeros((M, M))
    denominador = P_c + (1 / rho) * np.sum(p_km)
    for k in range(M):
        for m in range(M):
            selected_g_ru = np.random.choice(g_ru)
            numerador = W * np.log2(1 + (p_km[k, m] * g_t * selected_g_ru * L_k[k]) / (I_ki[k, m] + I_d[m] + N_0 * W))
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
def otimizar_iterativamente(p_e, P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, distancias, N_0, max_iter=10, tol=1e-3):
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
            selected_g_ru = np.random.choice(g_ru)
            sinc_term = sinc(f_k[k] * T_s)
            I_d_km[k, m] = p_km[k, m] * g_t * selected_g_ru * L_k[k] * (1 - sinc_term**2)

    return I_d_km


# Eq.3 (Modelo Global)
def calculate_intrauser_interference(p_km, g_s, g_ru, L_k, X):

    K, M = p_km.shape  # K é o número de usuários, M é o número de potências de transmissão
    I_i_km = np.zeros((K, M))  # Inicializar a matriz de interferência intrausuário com zeros
    
    for k in range(K):
        for m in range(M):
            interference_sum = 0
            for k_prime in range(K):
                selected_g_ru = np.random.choice(g_ru)
                for m_prime in range(M):
                    if k_prime != k or m_prime != m:
                        interference_sum += p_km[k_prime, m_prime] * X[k_prime, m_prime]
            I_i_km[k, m] = g_s * selected_g_ru * L_k[k] * interference_sum

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
            selected_g_ru = np.random.choice(g_ru)
            numerator = p_km[k, m] * g_t * selected_g_ru * L_k[k]
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




def algoritmo_1(P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, c, F_c, T_s, v, phi_k_list, usuarios, N_0, epsilon=1e-3):

    # Inicializações
    p_eq = np.min([P_f, P_T / M, np.min(P_r / (g_s * g_b * L_b * M))])
    p_0 = np.full(M, p_eq)
    i = 0

    distancias = calcular_distancias_satelite_para_pontos(coordenadas_lat_long, h, R)
    L_k = calcular_L_k(c, P_T, g_t, g_ru, F_c, distancias, L_b, M)
    f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)

    X_prev, p_prev = None, None

    while True:
        # Algoritmo 2: Beam assignment
        X, p_km, I_ki, q_km = otimizar_iterativamente(p_e, P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, distancias, N_0, max_iter=10, tol=1e-3)

        # Algoritmo 3: Alocação de potência
        p_star, lambda_n = dinkelbach_algorithm(p_0, c, P_T, g_t, g_ru, F_c, distancias, L_b, P_f, P_r, g_s, g_b, M, T_s, v, phi_k_list, usuarios, P_c, rho, N_0, W, epsilon=1e-5)

        # Critério de parada
        I_i_km = calculate_intrauser_interference(p_km, g_s, g_ru, L_k, X)
        I_d_km = calculate_interuser_interference(p_km, g_t, g_ru, L_k, f_k, T_s)

        gamma_km = calculate_snr(p_km, g_t, g_ru, L_k, I_i_km, I_d_km, N_0, W)
        if i > 0 and abs(eta_ef(X, p_star, P_c, rho, W, gamma_km) - eta_ef(X_prev, p_prev, P_c, rho, W, gamma_km)) < epsilon:
            break

        # Atualização para a próxima iteração
        p_0 = p_star
        X_prev, p_prev = X, p_star
        i += 1

    return p_star, X

# Executando o algoritmo 1 com os valores definidos
p_star, X_star = algoritmo_1(P_T, P_f, P_r, g_s, g_b, L_b, M, P_c, rho, W, g_t, g_ru, c, F_c, T_s, v, phi_k_list, usuarios, N_0, epsilon=1e-3)

print(f"Potências ótimas p*:\n{p_star}")
print(f"Matriz de alocação ótima X*:\n{X_star}")