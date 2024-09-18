import numpy as np

# Matriz A (coeficientes) y vector b (términos independientes)
A = np.array([[52, 30, 18],
              [20, 50, 30],
              [25, 20, 55]])

b = np.array([4800, 5810, 5690])

# Definir M y c según las fórmulas proporcionadas
M = np.array([[0, -30/18, -18/52],  # x1
              [-20/50, 0, -30/50],  # x2
              [-25/55, -20/55, 0]]) # x3

c = np.array([4800/52, 5810/50, 5690/55])

# Inicializamos las soluciones
x = np.zeros_like(b, dtype=np.double)

# Criterios de convergencia: cálculo de alfa
alfa_1 = abs(M[0, 1]) + abs(M[0, 2])
alfa_2 = abs(M[1, 0]) + abs(M[1, 2])
alfa_3 = abs(M[2, 0]) + abs(M[2, 1])
alfa_max = max(alfa_1, alfa_2, alfa_3)

# Mostrar los valores de alfa
print(f"Alfa 1: {alfa_1}")
print(f"Alfa 2: {alfa_2}")
print(f"Alfa 3: {alfa_3}")
print(f"Alfa máximo: {alfa_max}")

# Verificación de convergencia
if alfa_max <= 1:
    print("El sistema converge.")
else:
    print("El sistema no converge.")

# Definir una función para el método de Jacobi
def jacobi(M, c, tol=1e-10, max_iter=100):
    n = len(c)
    x_new = np.zeros_like(c)
    
    for k in range(max_iter):
        for i in range(n):
            suma = np.dot(M[i, :], x)
            x_new[i] = suma + c[i]

        # Comprobar convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Convergencia alcanzada en la iteración {k + 1}")
            return x_new

        x[:] = x_new  # Actualizar soluciones
    
    print("No se alcanzó la convergencia después del número máximo de iteraciones")
    return x_new

# Si el sistema converge, ejecutar el método de Jacobi
if alfa_max <= 1:
    soluciones = jacobi(M, c)
    print(f"Soluciones finales: x1 = {soluciones[0]}, x2 = {soluciones[1]}, x3 = {soluciones[2]}")
else:
    print("No se puede aplicar el método de Jacobi, ya que el sistema no converge.")
