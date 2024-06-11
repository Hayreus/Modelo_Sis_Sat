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
g_t = 10**(52.1/10)                  # Ganho da antena do satélite em dB
g_s = 10**(5/10)                      # Lóbulo lateral da antena de satélite em dB
g_k = [10, 11, 12, 13, 14, 15]      # Ganho da antena dos usuários, intervalo de 10 a 15 com passo de 0.5 em dB
g_b = 10**(5/10)                      # Ganho da estação de base em dB
P_f = 10**(10/10)                     # Potência máxima transmitida em dBw***
P_r = 10**(-111/10)         # Potência de interferência admissível em dBw
P_c = 10**(10/10)                    # Dissipação de potência do circuito em dBw
rho = 0.8                   # Eficiência do amplificador 
R = 6371                    # Raio médio da Terra em Km
xi = 15 * math.pi / 180     # Angulo minimo de elevação dado em graus e convertido para radianos
n =  7                     # Número de celulas hexagonais
N = 1                   # Número de niveis de camada hexagonais

# Dados de teste
L_b = 1                         # Perda de transmissão (ajuste conforme necessário)
P_T = 10**(30/10)                   # Potência total de transmissão em dBw
p = [10**(0.5/10), 10**(1/10), 10**(1.5/10), 10**(2/10), 10**(2.5/10), 10**(1.7/10), 10**(1.3/10)]     # Potência transmitida em cada feixe
g_ru = [10**(10/10), 10**(15/10), 10**(16/10), 10**(10/10), 10**(9/10), 10**(14/10), 10**(13/10)]         # Ganho da antena dos usuários em dB
L = [1e-3, 2e-3, 1.5e-3, 1e-3, 2e-3, 2e-3, 1.7e-3]  # Atenuação de percurso para cada feixe

num_usuario_por_celula = 1


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




def calcular_area_cobertura(R, psi):

    A = 2 * np.pi * R**2 * (1 - np.cos(psi / 2))
    return A

# Calcula a área de cobertura do feixe
Area_de_Cobertura = calcular_area_cobertura(R, psi)
print(f"--Área de Cobertura: {Area_de_Cobertura:.4f} km²")




def calcular_area_calota_esferica(R, psi):

    area_calota_esferica = 2 * math.pi * R ** 2 * (1 - math.cos(psi))
    return area_calota_esferica


# Calcular a área da calota esférica
area_calota_esferica = calcular_area_calota_esferica(R, psi)
print(f"--Área da calota esférica: {area_calota_esferica:.4f} km²")




def calcular_raio(Area_de_Cobertura):

    raio = math.sqrt(Area_de_Cobertura / math.pi)
    return raio


Raio_Total = calcular_raio(Area_de_Cobertura)
raio = (calcular_raio(Area_de_Cobertura))/2
diametro = Raio_Total*2
print(f"--Raio da área de cobertura da célula: {raio:.4f} km")
print(f"--Raio da área de cobertura Total: {Raio_Total:.4f} km")
print(f"--Diâmetro da área de cobertura Total: {diametro:.4f} km")




def calcular_comprimento_arco(R, psi):

    L = R * psi
    return L


# Calcular o comprimento do arco do setor circular
comprimento_arco = calcular_comprimento_arco(R, psi)
print(f"--Comprimento do arco do setor circular: {comprimento_arco:.4f} km")




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
    circulo_central = plt.Circle((0, 0), raio, edgecolor='b', facecolor='none', linestyle='--')
    ax.add_artist(circulo_central)
    
    pontos_central = gerar_pontos_no_circulo((0, 0), raio, num_pontos_por_circulo)
    todas_coordenadas.extend(pontos_central)
    for ponto in pontos_central:
        ax.plot(ponto[0], ponto[1], 'k.', alpha=0.5)
    
    # Adiciona os círculos ao redor e seus pontos
    pontos_hexagonais = calcular_pontos_hexagonais(raio)
    for centro in pontos_hexagonais:
        circulo = plt.Circle(centro, raio, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_artist(circulo)
        
        pontos = gerar_pontos_no_circulo(centro, raio, num_pontos_por_circulo)
        todas_coordenadas.extend(pontos)
        for ponto in pontos:
            ax.plot(ponto[0], ponto[1], 'k.', alpha=0.5)
    
    # Calcular o raio máximo para o círculo externo
    raio_maximo = 2 * raio + raio
    
    # Adiciona o círculo externo
    circulo_externo = plt.Circle((0, 0), raio_maximo, edgecolor='g', facecolor='none', linestyle='-')
    ax.add_artist(circulo_externo)
    
    # Configurações do gráfico
    ax.set_xlim(-4 * raio, 4 * raio)
    ax.set_ylim(-4 * raio, 4 * raio)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title("Distribuição de 7 Círculos com Pontos Aleatórios")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    return todas_coordenadas


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
print(f"Coordenadas em latitude e longitude (radianos): {coordenadas_lat_long}")




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