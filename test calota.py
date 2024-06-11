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

import math

def calcular_area_calota_esferica(R, psi):
    """
    Calcula a área da calota esférica com um ângulo central dado.
    
    Parâmetros:
    R (float): Raio da esfera.
    psi (float): Ângulo central da calota esférica em radianos.
    
    Retorna:
    float: Área da calota esférica.
    """
    area_calota_esferica = 2 * math.pi * R ** 2 * (1 - math.cos(psi))
    return area_calota_esferica


Area_de_Cobertura = calcular_area_calota_esferica(R, psi)
print(f"--Área de Cobertura: {Area_de_Cobertura:.2f} km²")


def calcular_comprimento_arco(R, psi):
    """
    Calcula o comprimento do arco de um setor circular de raio R e ângulo central psi em radianos.

    Args:
    - R (float): Raio do setor circular.
    - psi (float): Ângulo central do setor em radianos.

    Returns:
    - L (float): Comprimento do arco do setor circular.
    """
    L = R * psi
    return L


# Calcular o comprimento do arco do setor circular
comprimento_arco = calcular_comprimento_arco(R, psi)
print("--Comprimento do arco do setor circular:", comprimento_arco, "km")


#Eq. 2 (Posição Global)
def calcular_beta(psi, n):
    beta = (2 * psi) / (2 * n + 1)
    
    return beta
beta = calcular_beta(psi, n)
print(f"--Angulo de cobertura de cada célula: {beta}")


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
print(f"--largura do feixe da célula central: {theta_0}")


#Eq. 5 (Posição Global)
def calcular_theta_n(R, h, beta, theta_0, N):
    """
    Calcula a largura do feixe da enésima coroa de acordo com a equação fornecida.

    Args:
        R (float): Raio médio da Terra.
        h (float): Altitude do satélite.
        beta (float): Ângulo de cobertura de cada célula em radianos.
        theta_0 (float): Largura do feixe da célula central em radianos.
        N (int): Número de coroas hexagonais.

    Returns:
        numpy.ndarray: Largura do feixe para cada coroa de 0 a N em radianos.
    """
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

print(f"--Abertura θ das células = {theta_n} radianos")





def calcular_raio(Area_de_Cobertura):
    """
    Calcula o raio de um círculo a partir da área.
    
    Parâmetros:
    area (float): A área do círculo.
    
    Retorna:
    float: O raio do círculo.
    """
    raio = math.sqrt(Area_de_Cobertura / math.pi)
    return raio


raio = calcular_raio(Area_de_Cobertura)
print(f"--Raio da área de cobertura: {raio:.4f} km")




def calcular_pontos_hexagonais(raio):
    """
    Calcula as coordenadas dos centros dos 6 círculos ao redor do círculo central.
    
    Parâmetros:
    raio (float): O raio dos círculos.
    
    Retorna:
    list: Uma lista de coordenadas (x, y) dos centros dos 6 círculos.
    """
    angulos = np.linspace(0, 2 * np.pi, 7)[:-1]  # Divide o círculo em 6 partes iguais
    pontos = [(2 * raio * np.cos(angulo), 2 * raio * np.sin(angulo)) for angulo in angulos]
    return pontos

def gerar_pontos_no_circulo(centro, raio, num_pontos):
    """
    Gera num_pontos aleatórios dentro de um círculo dado o centro e o raio.
    
    Parâmetros:
    centro (tuple): Coordenadas (x, y) do centro do círculo.
    raio (float): O raio do círculo.
    num_pontos (int): Número de pontos a serem gerados.
    
    Retorna:
    list: Uma lista de coordenadas (x, y) dos pontos dentro do círculo.
    """
    pontos = []
    for _ in range(num_pontos):
        r = raio * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        x = centro[0] + r * np.cos(theta)
        y = centro[1] + r * np.sin(theta)
        pontos.append((x, y))
    return pontos

def plotar_circulos_e_pontos(raio, num_pontos_por_circulo):
    """
    Plota 7 círculos com um círculo central e 6 ao redor dele, distribuindo pontos aleatórios em cada círculo.
    
    Parâmetros:
    raio (float): O raio dos círculos.
    num_pontos_por_circulo (int): Número de pontos a serem distribuídos em cada círculo.
    
    Retorna:
    list: Uma lista com as coordenadas de todos os pontos.
    """
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

num_usuario_por_celula = 7
coordenadas_todos_usuarios = plotar_circulos_e_pontos(raio, num_usuario_por_celula)
print(f"--Coordenada de Usuários: {coordenadas_todos_usuarios}")
print(f"--Número Total de Usuários: {len(coordenadas_todos_usuarios)}")



def converter_xy_para_lat_long(coordenadas_todos_usuarios):
    """
    Converte coordenadas x e y em latitude e longitude usando a projeção de Mercator.
    
    Parâmetros:
    coordenadas (list): Lista de tuplas contendo coordenadas x e y em quilômetros.
    
    Retorna:
    list: Lista de tuplas contendo as latitudes e longitudes correspondentes para cada ponto.
    """
    
    # Fator de conversão de graus para radianos
    deg_to_rad = np.pi / 180.0
    
    # Lista para armazenar as latitudes e longitudes calculadas
    coordenadas_lat_long = []
    
    # Itera sobre cada ponto e calcula a latitude e longitude correspondentes
    for ponto in coordenadas_todos_usuarios:
        x, y = ponto
        # Conversão de coordenadas x e y para latitude e longitude
        latitude = np.arcsin(y / R) / deg_to_rad
        longitude = np.arctan2(x, R * np.cos(latitude * deg_to_rad)) / deg_to_rad
        
        coordenadas_lat_long.append((latitude, longitude))
    
    return coordenadas_lat_long

# Exemplo de uso
coordenadas_lat_long = converter_xy_para_lat_long(coordenadas_todos_usuarios)
print("--Coordenadas de latitude e longitude:")
print(coordenadas_lat_long)



Rs=h


def calcular_distancias_satelite_para_pontos(coordenadas_lat_long, Rs, R, phi):
    """
    Calcula as distâncias entre o satélite e cada ponto especificado pelas coordenadas de latitude e longitude.
    
    Parâmetros:
    coordenadas_lat_long (list): Lista de tuplas contendo as coordenadas de latitude e longitude de cada ponto.
    Rs (float): Raio geoestacionário (km).
    Re (float): Raio da Terra (km).
    phi (float): Latitude da estação terrestre (graus).
    
    Retorna:
    list: Lista das distâncias entre o satélite e cada ponto especificado.
    """
    # Converter a latitude de graus para radianos
    phi_rad = np.radians(phi)
    
    # Lista para armazenar as distâncias para cada ponto
    distancias = []
    
    # Coordenadas da estação terrestre
    Re_cos_phi = R * np.cos(phi_rad)
    Re_sin_phi = R * np.sin(phi_rad)
    
    # Calcular a distância para cada ponto
    for lat, long in coordenadas_lat_long:
        # Converter a latitude e longitude de graus para radianos
        lat_rad = np.radians(lat)
        long_rad = np.radians(long)
        
        # Coordenadas do ponto
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_long_diff = np.cos(long_rad)
        sin_long_diff = np.sin(long_rad)
        
        # Calcular a distância usando a equação fornecida
        d2 = (Rs - Re_cos_phi * cos_long_diff * cos_lat) ** 2 + (Re_sin_phi * cos_long_diff * cos_lat) ** 2 + (Re_sin_phi * sin_lat) ** 2
        distancia = np.sqrt(d2)
        distancias.append(distancia)
    
    return distancias

# Exemplo de uso

phi = 0     # Latitude da estação terrestre em graus

distancias = calcular_distancias_satelite_para_pontos(coordenadas_lat_long, Rs, R, phi)
print(f"--Distâncias do satélite para cada ponto: {distancias} Km")
