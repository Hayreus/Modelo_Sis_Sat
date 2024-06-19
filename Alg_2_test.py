import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import linprog
from munkres import Munkres, print_matrix

# Parâmetros
N = 1                   # Número de niveis de camada hexagonais
n =  7                     # Número de celulas hexagonais
num_usuario_por_celula = 15
usuarios = num_usuario_por_celula * n
c = 299792458               # Velocidade da luz no vácuo em m/s
h = 780                     # Altitude Orbital em km
v = 7460                    # Velocidade orbital em m/s
F_c = 20e9                    # Frequencia de centro em hz
W = 28e6              # Largura de banda em MHz 28 (28e6 Hz)
T_s = 1                     # Tempo de duração do símbolo em micro segundo
micro = -2.6                # Parâmetro de desvanecimento da chuva em dB*
sigma =  1.6                # Parâmetro de desvanecimento da chuva em dB*
N_0 = 10**(-172/10)         # Densidade espetral do ruído em dBw/Hz para W/Hz
M = usuarios                       # Número de feixes
g_t = 10**(52.1/10)                  # Ganho da antena do satélite em W
g_s = 10**(5/10)                     # Lóbulo lateral da antena de satélite em dB
g_k = [10**(10/10), 10**(11/10), 10**(12/10), 10**(13/10), 10**(14/10), 10**(15/10)]       # Ganho da antena dos usuários, intervalo de 10 a 15 com passo de 0.5 em dB
g_b = 10**(5/10)                     # Ganho da estação de base em w
P_f = 10**(10/10)                    # Potência máxima transmitida em w
P_r = 10**(-111/10)         # Potência de interferência admissível em w
P_c = 10**(10/10)           # Dissipação de potência do circuito em dBw
rho = 0.8                   # Eficiência do amplificador 
R = 6371                    # Raio médio da Terra em Km
xi = 15 * math.pi / 180     # Angulo minimo de elevação dado em graus e convertido para radianos


# Dados de teste
L_b = 1                         # Perda de transmissão (ajuste conforme necessário)
P_T = 10**(30/10)                   # Potência total de transmissão em W
p = [10**(0.5/10), 10**(1/10), 10**(1.5/10), 10**(2/10), 10**(2.5/10), 10**(1.7/10), 10**(1.3/10)]     # Potência transmitida em cada feixe
g_ru = [10**(10/10), 10**(15/10), 10**(16/10), 10**(10/10), 10**(9/10), 10**(14/10), 10**(13/10)]         # Ganho da antena dos usuários em dB
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

print(f"--Abertura θ das células: {theta_n} radianos")




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
print(f"--Coordenada de Usuários: {coordenadas_todos_usuarios}")
print(f"--Número Total de Usuários: {len(coordenadas_todos_usuarios)}")




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
print("Coordenadas em latitude e longitude (radianos):", coordenadas_lat_long)




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


# Calcular distâncias
distancias = calcular_distancias_satelite_para_pontos(coordenadas_lat_long, h, R)
print(f"Distâncias para cada ponto: {distancias} Km")




############################################################################################




# Equações para algoritmo 2 e 3

def calcular_L_k(c, P_T, g_t, g_k, F_c, distancias, L_b, usuarios):

    lambda_ = c / F_c  # Comprimento de onda
    K = usuarios
    L_k = np.zeros(K)
    
    for k in range(K):
        g_k_m = np.random.choice(g_k)  # Seleciona aleatoriamente um valor de g_k da lista
        d_m = distancias[k] * 1000  # Converte a distância de km para m
        d_k = P_T * (g_t * g_k_m * lambda_**2) / ((4 * np.pi * d_m)**2 * L_b)
        h_k = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)  # Variável aleatória complexa
        L_k[k] = d_k * np.abs(h_k)**2
    
    return L_k


L_k = calcular_L_k(c, P_T, g_t, g_k, F_c, distancias, L_b, usuarios)
print(f"--Potência recebida a uma distância (em watts, W): {L_k}")




#Eq. 5 (Modelo Global)
def calcular_phi_k(coordenadas_lat_long, lat_sat=0, lon_sat=0):
    """
    Calcula os valores de phi_k para cada ponto de uma lista de coordenadas.

    Parâmetros:
    - coordenadas_lat_long (list): Lista de tuplas contendo as coordenadas de latitude e longitude de cada ponto em radianos.
    - lat_sat (float): Latitude do satélite em radianos (padrão: 0).
    - lon_sat (float): Longitude do satélite em radianos (padrão: 0).

    Retorna:
    - phi_k_list (numpy array): Lista contendo os valores calculados de phi_k para cada ponto.
    """
    
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
print(f"--phi_k para cada ponto: {phi_k_list} radianos")




# Função para calcular f_k
def calcular_fk(v, F_c, c, phi_k_list, usuarios):

    K = usuarios
    f_k = []
    for k in range(K):
        angle = phi_k_list[k]
        f_k.append((v * F_c / c) * np.cos(angle))
    return f_k


# Calcular f_k
f_k = calcular_fk(v, F_c, c, phi_k_list, usuarios)
print(f"--Eq. 5 Frequência desviada associada ao k-ésimo usuário: {f_k}")




#Eq.34 (Modelo Global)
def calculate_pe(P_f, P_T, P_r, g_s, g_b, L_b, M):

    pe1 = P_f
    pe2 = P_T / M
    pe3 = P_r / (g_s * g_b * L_b * M)
    
    # Calcula o valor mínimo entre os três candidatos
    pe0 = min(pe1, pe2, pe3)
    
    return pe0


def optimize_energy_efficiency(K, M, P_T, P_f, P_r, g_s, g_b, L_b):
    """
    Resolve o problema de otimização de eficiência energética para sistemas de comunicação por satélite.
    
    Args:
    K (int): Número de usuários.
    M (int): Número de feixes.
    P_T (float): Potência total disponível.
    P_f (float): Potência máxima por feixe.
    P_r_p (float): Limite de interferência.
    g_s (float): Ganho da antena do satélite.
    g_b (float): Ganho da antena da base.
    L_b (float): Atenuação.
    
    Returns:
    dict: Resultados da otimização.
    """
    # Calcula p_e0
    p_e0 = calculate_pe(P_f, P_T, P_r, g_s, g_b, L_b, M)
    
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
            "p_e0": p_e0,
            "fun": -result.fun  # Inverter o sinal de volta, pois minimizamos -η
        }
    else:
        return {
            "success": result.success,
            "message": result.message
        }


p_e = calculate_pe(P_f, P_T, P_r, g_s, g_b, L_b, M)
print(f"--Eq.34 Valores iniciais de potência: {p_e}")




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
print(f"--Eq.21 Interferência de outras fontes: {I_d}")


#Eq.20 (Modelo Global)
def calcular_I_i(p, g_s, g_ru, L):
    I_i = [0] * len(p)
    
    for k in range(len(p)):
        # Calcula a interferência interna para o feixe k
        I_i[k] = g_s * g_ru[k] * L[k] * sum(p[k_prime] for k_prime in range(len(p)) if k_prime != k)
    
    return I_i
I_i = calcular_I_i(p, g_s, g_ru, L)
print(f"--Eq.20 Lista de interferências internas para cada feixe: {I_i}")


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
print(f"--Eq.19 Eficiência Energética Calculada (W): {eta}")


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

print("--Eq. 22 Potências ótimas dos feixes:", p_star)
print("--Eq. 22 Valor máximo da eficiência energética:", p_max)


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

print(f"--Eq.27 Valor de f1: {f_1}")
print(f"--Eq.28 Valor de f2: {f_2}")


#Eq.26 (Modelo Global)
def calcular_C_p(p, f_1, f_2, W):
    C_p = np.zeros(len(p))
    for k in range(len(p)):
        C_p[k] = W * (f_1[k] - f_2[k])
    return C_p
C_p = calcular_C_p(p, f_1, f_2, W)
print(f"--Eq.26 Soma ponderada das diferenças entre f1(p_k) e f2(p_k): {C_p}")


#Eq.25 (Modelo Global)
def grad_f2(I_i, I_d, N_0, W):
    K = len(I_i)  # Número de feixes
    grad_values = np.zeros(K)
    
    for k in range(K):
        interference = I_i[k] + I_d[k] + N_0 * W
        grad_values[k] = 1 / (np.log(2) * interference)  # Derivada de log2(interference)
    
    return grad_values
    
gra_f2 = grad_f2(I_i, I_d, N_0, W)
print(f"--Eq.25 Gradiente de f2: {gra_f2}")

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
print(f"--Eq.24 Capacidade de transmissão total no ponto ótimo: {tilde_C_p_star}")


#Eq.23 (Modelo Global)
def calcular_D(p_star, P_c, rho):
    D_p_star = P_c + (1 / rho) * sum(p_star)
    return D_p_star

D_p_star = calcular_D(p_star, P_c, rho)
print(f"--Eq.23 D_p_star: {D_p_star}")

def calcular_lambda_estrela(tilde_C_p_star, D_p_star):
    # Calcula lambda* como a razão entre a capacidade de transmissão total e a potência total consumida
    lambda_estrela = tilde_C_p_star / D_p_star
    return lambda_estrela

lambda_estrela = calcular_D(p_star, P_c, rho)
print(f"--Eq.23 Eficiência energética máxima alcançável no sistema: {lambda_estrela}")


#Eq.29 (Modelo Global)
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
print(f"--Eq.29 Resultado da maximização: {resultado_max}")



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

print(f"--Alg. 3 Potências ótimas dos feixes: {p_star}")
print(f"--Alg. 3 Eficiência energética máxima alcançável no sistema: {lambda_star}")



####################################################################################
# Equações para algoritmo 2


# Eq. 12 (Modelo Global)
def calcular_p_km(P_f, P_T, P_r, g_s, g_b, L_b, M):

    K = n
    p_km = np.zeros((K, M))

    for k in range(K):
        for m in range(M):
            p_km[k, m] = min(P_f, P_T / M, P_r / (g_s * g_b * L_b * M))
    
    return p_km

p_km = calcular_p_km(P_f, P_T, P_r, g_s, g_b, L_b, M)
print(f"p_km: \n{p_km}")