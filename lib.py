##POSIÇÃO GLOBAL

def calcular_psi(R, h, eta):
    termo1 = 2 * R * (math.pi / 2)
    termo2 = 2 * R * (-eta)
    termo3 = 2 * R * math.asin((R / (R + h)) * math.cos(eta))

    psi = termo1 + termo2 + termo3
    return psi

def calcular_beta(psi, n):
    # Calcula beta usando a fórmula fornecida
    beta = (2 * psi) / (2 * n + 1)
    return beta

def calcular_Nc(n):
    # Calcula Nc usando a fórmula fornecida
    Nc = 1 + (6 * n * (n + 1)) / 2
    return Nc

def calcular_theta_0_2(R, h, beta):
    # Calcula theta_0/2 usando a fórmula fornecida
    numerador = R * math.sin(beta / 2)
    denominador = h + R - R * math.cos(beta / 2)
    
    theta_0_2 = math.atan2(numerador, denominador)
    return theta_0_2

def calcular_theta_n(R, h, beta, n):
    # Termo 1: R * sin((2n+1)beta/2)
    termo1 = R * math.sin((2 * n + 1) * beta / 2)

    # Termo 2: h + R - R * cos((2n+1)beta/2)
    termo2 = h + R - R * math.cos((2 * n + 1) * beta / 2)

    # Calcula theta_n usando a fórmula fornecida
    theta_n = math.atan2(termo1, termo2)

    # Calcula a soma dos termos anteriores: Σθ_k
    soma_theta_k = sum(math.atan2(R * math.sin((2 * k + 1) * beta / 2), h + R - R * math.cos((2 * k + 1) * beta / 2)) for k in range(1, n))

    # Termo 3: θ_0/2
    termo3 = calcular_theta_0_2(R, h, beta)

    # Subtrai os termos anteriores da fórmula principal
    theta_n -= (soma_theta_k + termo3)

    return theta_n

##MODELO GLOBAL

def calcular_gs_gt(theta, delta):
    # Calcula g_t usando a primeira equação
    gt = (2 * math.pi - (2 * math.pi - theta) * delta) / theta

    # Calcula g_s usando a segunda equação
    gs = delta

    return gs, gt

def calcular_gamma(p_km, gt, g_ru_k, L_k, I_i_k_m, I_d_k_m, N_0, W):
    # Calcula gamma_k_m usando a fórmula fornecida
    gamma_k_m = (p_km * gt * g_ru_k * L_k) / (I_i_k_m + I_d_k_m + N_0 * W)
    
    return gamma_k_m

def calcular_I_i_k_m(gs, g_ru_k, L_k, p, x):
    # Inicializa o valor de I_i_k_m
    I_i_k_m = 0

    # Loop sobre todos os índices m' e k'
    for m_prime in range(len(x)):
        for k_prime in range(len(x[0])):
            if m_prime != m and k_prime != k:
                # Adiciona o termo ao somatório
                I_i_k_m += p[k_prime][m_prime] * x[k_prime][m_prime]

    # Multiplica pelos fatores restantes da equação
    I_i_k_m *= gs * g_ru_k * L_k

    return I_i_k_m

def calcular_I_d_k_m(p_km, gt, g_ru_k, L_k, f_k, T_s):
    # Calcula I_d_k_m usando a fórmula fornecida
    sinc_term = math.sin(math.pi * f_k * T_s) / (math.pi * f_k * T_s) if f_k * T_s != 0 else 1
    I_d_k_m = p_km * gt * g_ru_k * L_k * (1 - sinc_term**2)

    return I_d_k_m

def calcular_f_k(v, f_c, c, phi_k):
    # Calcula f_k usando a fórmula fornecida
    f_k = (v * f_c / c) * math.cos(phi_k)

    return f_k

def calcular_R_k(gamma, W, x):
    # Inicializa o valor de R_k
    R_k = 0

    # Loop sobre todos os índices m
    for m in range(len(x[0])):
        # Adiciona o termo ao somatório
        R_k += W * math.log2(1 + gamma[m]) * x[m]

    return R_k

def calcular_Pa(p, x):
    # Inicializa o valor de Pa
    Pa = 0

    # Loop sobre todos os índices k e m
    for k in range(len(x)):
        for m in range(len(x[0])):
            # Adiciona o termo ao somatório
            Pa += p[k][m] * x[k][m]

    # Divide pelo fator de escala (1/ρ)
    Pa /= len(x) * len(x[0])

    return Pa

def calcular_Ptot(Pc, Pa):
    # Calcula Ptot usando a fórmula fornecida
    Ptot = Pc + Pa
    return Ptot

def calcular_Ib(gs, gb, Lb, p, x):
    # Inicializa o valor de Ib
    Ib = 0

    # Loop sobre todos os índices k e m
    for k in range(len(x)):
        for m in range(len(x[0])):
            # Adiciona o termo ao somatório
            Ib += x[k][m] * p[k][m]

    # Multiplica pelos fatores restantes da equação
    Ib *= gs * gb * Lb

    return Ib