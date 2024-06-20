from scipy.optimize import linear_sum_assignment
import numpy as np

# Função para otimização de X
def otimizar_X(q_km, K, M):
    # Convertendo o problema de maximização para minimização
    cost_matrix = -q_km
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    X = np.zeros((K, M))
    X[row_ind, col_ind] = 1
    return X

# Parâmetros de exemplo
K = 10  # Número de usuários (exemplo)
M = 10  # Número de subportadoras (exemplo)
q_km = np.random.rand(K, M)  # Matriz de exemplo com valores aleatórios

# Otimização de X
X = otimizar_X(q_km, K, M)

print("Matriz de alocação X:")
print(X)
