import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
import time

def armijo_rule(f, gradient, x, direction, alpha=0.2, beta=0.5):

    t = 1.0  # Tamanho de passo inicial
    fun = f(x + t * direction)
    funaprox = f(x) + alpha * t * np.dot(gradient, direction)
    count = 0
    while (fun > funaprox) or (np.isnan(fun)):
        t *= beta
        fun = f(x + t * direction)
        funaprox = f(x) + alpha * t * np.dot(gradient, direction)
        count += 1
    return t, count

def approximate_gradient(f, x, epsilon=1e-6):
    """
    Aproxima o gradiente de uma função em um ponto usando a diferença finita.

    Parâmetros:
    - f: Função para a qual o gradiente será aproximado.
    - x: Ponto no qual o gradiente será aproximado (vetor).
    - epsilon: Pequeno valor para calcular a diferença finita.

    Retorna:
    - A aproximação do gradiente em x (vetor).
    """
    num_variables = len(x)
    gradient_approximation = np.zeros(num_variables)

    for i in range(num_variables):
        x_plus_epsilon = x.copy()
        x_plus_epsilon[i] += epsilon

        x_minus_epsilon = x.copy()
        x_minus_epsilon[i] -= epsilon

        partial_derivative_approximation = (f(x_plus_epsilon) - f(x_minus_epsilon)) / (2 * epsilon)
        gradient_approximation[i] = partial_derivative_approximation

    return gradient_approximation

def ana_gradient(x):
    gradient = np.zeros(5)
    gradient[0] = (np.log(x[0]) + 1)/(x[0] * np.sqrt((np.log(x[0]) + 1)**2 + (np.log(x[1]) + 1)**2 + np.log(x[2] + x[3] + x[4] + 0.1)**4))
    gradient[1] = (np.log(x[1]) + 1)/(x[1] * np.sqrt((np.log(x[0]) + 1)**2 + (np.log(x[1]) + 1)**2 + np.log(x[2] + x[3] + x[4] + 0.1)**4))
    gradient[2] = 2*(np.log(x[2] + x[3] + x[4] + 0.1)**3)/((x[2] + x[3] + x[4] + 0.1) * np.sqrt((np.log(x[0]) + 1)**2 + (np.log(x[1]) + 1)**2 + np.log(x[2] + x[3] + x[4] + 0.1)**4))
    gradient[3] = 2*(np.log(x[2] + x[3] + x[4] + 0.1)**3)/((x[2] + x[3] + x[4] + 0.1) * np.sqrt((np.log(x[0]) + 1)**2 + (np.log(x[1]) + 1)**2 + np.log(x[2] + x[3] + x[4] + 0.1)**4))
    gradient[4] = 2*(np.log(x[2] + x[3] + x[4] + 0.1)**3)/((x[2] + x[3] + x[4] + 0.1) * np.sqrt((np.log(x[0]) + 1)**2 + (np.log(x[1]) + 1)**2 + np.log(x[2] + x[3] + x[4] + 0.1)**4))
    
    return gradient

def gradient_descent_multivariable(f, initial_guess, num_iterations=1000):
    
    x = initial_guess
    armijo_count = 0

    for i in range(num_iterations):
        gradient = approximate_gradient(f, x)
        t, count = armijo_rule(f, gradient, x, -1*gradient)
        x = x - t * gradient
        armijo_count += count
        

    return f(x), x, i+1, armijo_count

def Newton(f, initial_guess, num_iteractions=1000):
    x = initial_guess
    armijo_count = 0

    for i in range(num_iteractions):
        gradient = approximate_gradient(f, x)
        hessian =  jacobian(egrad(f))(x)
        H = np.linalg.inv(hessian)
        d = (H) * gradient
        t, count = armijo_rule(f, gradient, x, -1*H)
        x = x - t*d
        armijo_count += count

    
    return f(x), x, i+1, armijo_count

def DFP_method(f, initial_guess, num_iteractions=1000):
    x = initial_guess
    H = np.identity(len(x))
    armijo_count = 0
    gradient = ana_gradient(x)

    for i in range(num_iteractions):
        d = H @ gradient
        t, count = armijo_rule(f, gradient, x, -1*d)
        armijo_count += count
        x_old = x
        x = x - t*d
        gradient_old = gradient
        gradient = ana_gradient(x)
        pk = x - x_old
        qk = gradient - gradient_old
        #H = H + (np.outer(pk, pk))/(np.dot(pk, qk)) - (np.outer((H @ qk), qk) @ H)/np.dot(qk @ H, qk)
        
    
    return f(x), x, i+1, armijo_count


def ofunction(x):
    return np.sqrt((np.log(x[0]) + 1)**2 + (np.log(x[1]) + 1)**2 + np.log(x[2] + x[3] + x[4] + 0.1)**4)
    

# Ponto no qual o gradiente será aproximado
x_values = np.array([15.0, 5.0, 2.0, 3.0, 7.0])



#min_value, min_point, iteractions, armijo = gradient_descent_multivariable(ofunction, x_values)
#min_value, min_point, iteractions = Newton(ofunction, x_values)
current_time = time.time()
min_value, min_point, iteractions, armijo = DFP_method(ofunction, x_values)
end_time = time.time()

print(f"Valor mínimo: {min_value}")
print(f"Ponto correspondente: {min_point}")
print(f"numereo de iterações: {iteractions}")
print(f"numero de chamadas de Armijo: {armijo}")
print(f"Tempo de execucao: {end_time - current_time}")
