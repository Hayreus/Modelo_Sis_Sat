import numpy as np
import matplotlib.pyplot as plt

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

# Parâmetros
raio = 868.4164  # Raio dos círculos em km
num_usuario_por_celula = 7  # Número de pontos por círculo

# Plotar círculos e pontos
coordenadas_todos_usuarios = plotar_circulos_e_pontos(raio, num_usuario_por_celula)
print(f"--Coordenada de Usuários: {coordenadas_todos_usuarios}")
print(f"--Número Total de Usuários: {len(coordenadas_todos_usuarios)}")
