#include <iostream>
#include <vector>
#include <numeric>
#include <windows.h>
#include <math.h>
#include <iomanip>
#include <stdexcept>

double xSquare(std::vector<double> x, int length) {
    double result;

    result = sqrt(pow(log(x[0]) + 1, 2) + pow(log(x[1]) + 1, 2) + pow(log(x[2] + x[3] + x[4] + 0.1), 4));
    
    return result;
}

std::vector<double> vectorSub(std::vector<double> vec1, std::vector<double> vec2, int length) {
    std::vector<double> resultado(length, 0);

    for(int i = 0; i < length; i++) {
        resultado[i] = vec1[i] - vec2[i];
    }

    return resultado;
}

double getDeterminant(const std::vector<std::vector<double>> vect) {
    if(vect.size() != vect[0].size()) {
        throw std::runtime_error("Matrix is not quadratic");
    } 
    int dimension = vect.size();

    if(dimension == 0) {
        return 1;
    }

    if(dimension == 1) {
        return vect[0][0];
    }

    //Formula for 2x2-matrix
    if(dimension == 2) {
        return vect[0][0] * vect[1][1] - vect[0][1] * vect[1][0];
    }

    double result = 0;
    int sign = 1;
    for(int i = 0; i < dimension; i++) {

        //Submatrix
        std::vector<std::vector<double>> subVect(dimension - 1, std::vector<double> (dimension - 1));
        for(int m = 1; m < dimension; m++) {
            int z = 0;
            for(int n = 0; n < dimension; n++) {
                if(n != i) {
                    subVect[m-1][z] = vect[m][n];
                    z++;
                }
            }
        }

        //recursive call
        result = result + sign * vect[0][i] * getDeterminant(subVect);
        sign = -sign;
    }

    return result;
}

std::vector<std::vector<double>> getTranspose(const std::vector<std::vector<double>> matrix1) {

    //Transpose-matrix: height = width(matrix), width = height(matrix)
    std::vector<std::vector<double>> solution(matrix1[0].size(), std::vector<double> (matrix1.size()));

    //Filling solution-matrix
    for(size_t i = 0; i < matrix1.size(); i++) {
        for(size_t j = 0; j < matrix1[0].size(); j++) {
            solution[j][i] = matrix1[i][j];
        }
    }
    return solution;
}

std::vector<std::vector<double>> getCofactor(const std::vector<std::vector<double>> vect) {
    if(vect.size() != vect[0].size()) {
        throw std::runtime_error("Matrix is not quadratic");
    } 

    std::vector<std::vector<double>> solution(vect.size(), std::vector<double> (vect.size()));
    std::vector<std::vector<double>> subVect(vect.size() - 1, std::vector<double> (vect.size() - 1));

    for(std::size_t i = 0; i < vect.size(); i++) {
        for(std::size_t j = 0; j < vect[0].size(); j++) {

            int p = 0;
            for(size_t x = 0; x < vect.size(); x++) {
                if(x == i) {
                    continue;
                }
                int q = 0;

                for(size_t y = 0; y < vect.size(); y++) {
                    if(y == j) {
                        continue;
                    }

                    subVect[p][q] = vect[x][y];
                    q++;
                }
                p++;
            }
            solution[i][j] = pow(-1, i + j) * getDeterminant(subVect);
        }
    }
    return solution;
}

std::vector<std::vector<double>> getInverse(const std::vector<std::vector<double>> vect) {
    if(getDeterminant(vect) == 0) {
        throw std::runtime_error("Determinant is 0");
    } 

    double d = 1.0/getDeterminant(vect);
    std::vector<std::vector<double>> solution(vect.size(), std::vector<double> (vect.size()));

    for(size_t i = 0; i < vect.size(); i++) {
        for(size_t j = 0; j < vect.size(); j++) {
            solution[i][j] = vect[i][j]; 
        }
    }

    solution = getTranspose(getCofactor(solution));

    for(size_t i = 0; i < vect.size(); i++) {
        for(size_t j = 0; j < vect.size(); j++) {
            solution[i][j] *= d;
        }
    }

    return solution;
}

std::vector<std::vector<double>> createId(int length) {
    std::vector<std::vector<double>> id(length, std::vector<double> (length, 0));
    
    for(int i = 0; i < length; i++) {
        id[i][i] = 1;
    }

    return id;
}

void printMatrix(const std::vector<std::vector<double>> vect) {
    for(std::size_t i = 0; i < vect.size(); i++) {
        for(std::size_t j = 0; j < vect[0].size(); j++) {
            std::cout << std::setw(8) << vect[i][j] << " ";
        }
        std::cout << "\n";
    }
}

double toDerivate(double (*fun)(std::vector<double> x, int length), double inf, std::vector<double> point, int length, int variable) {
    std::vector<double> diferencial = point;
    double derivative;

    for (int i = 0; i < length; i++) {
        if (i == variable) diferencial[i] = point[i] + inf;
    };

    derivative = (fun(diferencial, length) - fun(point, length))/inf;

    return derivative;
}

std::vector<double> gradient(double (*fun)(std::vector<double> x, int length), std::vector<double> x, int length, double inf) {
    /* Calculando o gradiente analiticamente*/
    std::vector<double> gradiente(length, 0);

    gradiente[0] = (log(x[0]) + 1)/ (x[0]*sqrt(pow(log(x[0]) + 1, 2) + pow(log(x[1]) + 1, 2) + pow(log(x[2] + x[3] + x[4] + 0.1), 4)));
    gradiente[1] = (log(x[1]) + 1)/ (x[1]*sqrt(pow(log(x[0]) + 1, 2) + pow(log(x[1]) + 1, 2) + pow(log(x[2] + x[3] + x[4] + 0.1), 4)));
    gradiente[2] = 2 * pow(log(x[2] + x[3] + x[4] + 1), 2)/ ((x[2]+x[3]+x[4]+1)*sqrt(pow(log(x[0]) + 1, 2) + pow(log(x[1]) + 1, 2) + pow(log(x[2] + x[3] + x[4] + 0.1), 4)));
    gradiente[3] = 2 * pow(log(x[2] + x[3] + x[4] + 1), 2)/ ((x[2]+x[3]+x[4]+1)*sqrt(pow(log(x[0]) + 1, 2) + pow(log(x[1]) + 1, 2) + pow(log(x[2] + x[3] + x[4] + 0.1), 4)));
    gradiente[4] = 2 * pow(log(x[2] + x[3] + x[4] + 1), 2)/ ((x[2]+x[3]+x[4]+1)*sqrt(pow(log(x[0]) + 1, 2) + pow(log(x[1]) + 1, 2) + pow(log(x[2] + x[3] + x[4] + 0.1), 4)));

    return gradiente;
}

double findStep(double (*fun)(std::vector<double> x, int length), std::vector<double> direction, std::vector<double> gradient, 
                std::vector<double> point, double ni, double initialStep, int length) {
    /* Calcula o tamanho do passo pelo criterio de Armijo */

    std::vector<double> nextPoint = point;
    double funValue, funAproxValue, step = initialStep, gradDotDirection;

    for(int i = 0; i < length; i++) {
        nextPoint[i] += step * direction[i];
    };

    gradDotDirection = std::inner_product(gradient.begin(), gradient.end(), direction.begin(), 0.0);

    funValue = fun(nextPoint, length);
    funAproxValue = fun(point, length) + ni*step*gradDotDirection;

    while((std::isnan(funAproxValue)) || (std::isnan(funValue)) || (funValue > funAproxValue)) {
        step = step * 0.8;
        nextPoint = point;

        for(int i = 0; i < length; i++) {
            nextPoint[i] += step * direction[i];
        };

        funValue = fun(nextPoint, length);
        funAproxValue = fun(point, length) + ni*step*gradDotDirection;
    }

    return step;
}

std::vector<double> matrixDotVector(std::vector<std::vector<double>> matrix, std::vector<double> vector, int length) {
    std::vector<double> vectorMatrix(length, 0), resultado(length, 0);

    for(int i = 0; i < length; i++) {
        vectorMatrix = matrix[i];
        
        for(int j = 0; j < length; j++) {
            resultado[i] += vectorMatrix[j] * vector[j];
        }
    }

    return resultado;
}

int isZero(std::vector<double> grad, int length, double tol) {
    /* Verifica se o vetor gradiente = 0 
    baseado em uma tolerancia */
    
    for (int i = 0; i < length; i++) {
        if ((grad[i] < tol)) continue;
        else return 0;
    }

    return 1;
}

std::vector<double> gradient_method(double (*fun)(std::vector<double> x, int length),  std::vector<double> initialPoint, double ni, int length) {
    std::vector<double> lastPoint, point = initialPoint, direction, grad;
    double step, n_iter = 0;

    grad = gradient(fun, point, length, 0.000001);

    while((isZero(grad, length, 0.01) != 1) && n_iter < 40) {
        direction = grad;

        for(int i = 0; i < length; i++) {
            direction[i] *= -1;
        };

        step = findStep(fun, direction, grad, point, 0, 1, length);
        
        for(int i = 0; i < length; i++) {
            point[i] += step*direction[i];
        };

        grad = gradient(fun, point, length, 0.000001);
        n_iter++;
    }

    std::cout << n_iter << '\n';

    return point;
}

double hessElement(double (*fun)(std::vector<double> x, int length), std::vector<double> point, int i, int j, int length, double inf) {
    double value, gradInf, grad;
    std::vector<double> pointInf = point;

    pointInf[j] += inf;
    gradInf = gradient(fun, pointInf, length, inf)[i];
    grad = gradient(fun, point, length, inf)[i];

    value = (gradInf - grad)/inf;

    return value; 
}

std::vector<std::vector<double>> createHess(double (*fun)(std::vector<double> x, int length), std::vector<double> point, int length, double inf) {
    std::vector<std::vector<double>> hess(length, std::vector<double>(length, 0));
    double hessValue;

    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            hessValue = hessElement(fun, point, i, j, length, inf);
            if (hessValue < 0.001) hess[i][j] = 0;
            else hess[i][j] = hessValue;
        }
    }

    return hess;
}

std::vector<double> newton_method(double (*fun)(std::vector<double> x, int length),  std::vector<double> initialPoint, double ni, int length) {
    std::vector<double> point = initialPoint, direction, grad;
    std::vector<std::vector<double>> hessiana(length, std::vector<double>(length, 0)), inverseHess(length, std::vector<double>(length, 0));
    double step;

    grad = gradient(fun, point, length, 0.000001);
    hessiana = createHess(fun, point, length, 0.00001);
    inverseHess = getInverse(hessiana);

    while(isZero(grad, length, 0.01) != 1) {
        direction = matrixDotVector(inverseHess, grad, length);
        for(int i = 0; i < length; i++) {
            direction[i] *= -1;
        };

        step = findStep(fun, direction, grad, point, ni, 1, length);
        
        for(int i = 0; i < length; i++) {
            point[i] += step*direction[i];
        };

        grad = gradient(fun, point, length, 0.000001);
        hessiana = createHess(fun, point, length, 0.0001);
        inverseHess = getInverse(hessiana);
    }

    return point;
}

std::vector<std::vector<double>> DFP_first_term(std::vector<double> p, std::vector<double> q, int length) {
    std::vector<std::vector<double>> firstMatrix(length, std::vector<double>(length, 0));
    double pXq;

    for(int i = 0; i < length; i++) {
        pXq += p[i] * q[i];
    }

    for(int i = 0; i < length; i++) {
        for(int j = 0; j < length; j++) {
            firstMatrix[i][j] = (p[i]*p[j])/pXq;
        }
    }

    return firstMatrix;
}

std::vector<std::vector<double>> DFP_sec_term(std::vector<std::vector<double>> H, std::vector<double> q, int length) {
    std::vector<std::vector<double>> matrix1(length, std::vector<double>(length, 0)), resultado(length, std::vector<double>(length, 0));
    std::vector<double> vectorDenom(length, 0), vectorMatrix, vec1(length, 0), vec2(length, 0);
    double denom;

    for(int i = 0; i < length; i++) {
        for(int j = 0; j < length; j++) {
            vectorDenom[i] += q[j] * H[j][i];
        }
    }

    for(int i = 0; i < length; i++) {
        denom += vectorDenom[i] * q[i];
    }

    for(int i = 0; i < length; i++) {
        vectorMatrix = H[i];
        
        for(int j = 0; j < length; j++) {
            vec1[i] += vectorMatrix[j] * q[j];
        }
    }

    for(int i = 0; i < length; i ++) {
        for(int j = 0; j < length; j++) {
            matrix1[i][j] = vec1[i]*q[j];
        }
    }

    for(int i = 0; i < length; i++) {
        vectorMatrix = matrix1[i];

        for(int j = 0; j < length; j++) {
            for(int k = 0; k < length; k++) {
                resultado[i][j] += (vectorMatrix[k] * H[k][j])/denom;
            }
        }
    }

    return resultado;
}

std::vector<std::vector<double>> next_H(std::vector<std::vector<double>> H, std::vector<std::vector<double>> first_term, std::vector<std::vector<double>> sec_term, int length) {
    std::vector<std::vector<double>> resultado(length, std::vector<double>(length, 0));

    for(int i = 0; i < length; i++) {
        for(int j = 0; j < length; j++) {
            resultado[i][j] = H[i][j] + first_term[i][j] - sec_term[i][j];
        }
    }

    return resultado;
}

std::vector<double> DFP_method(double (*fun)(std::vector<double> x, int length),  std::vector<double> initialPoint, double ni, int length) {
    std::vector<double> point = initialPoint, direction, grad, p(length, 0), q(length, 0), newGrad;
    std::vector<std::vector<double>> H = createId(length), nextH;
    double step, it_max = 0;

    grad = gradient(fun, point, length, 0.000001);

    while((isZero(grad, length, 0.01) != 1) && it_max < 8) {
        direction = matrixDotVector(H, grad, length);

        for(int i = 0; i < length; i++) {
            direction[i] *= -1;
        };

        step = findStep(fun, direction, grad, point, ni, 1, length);
        
        for(int i = 0; i < length; i++) {
            p[i] = step*direction[i];
            point[i] += step*direction[i];
        };

        newGrad = gradient(fun, point, length, 0.000001);
        q = vectorSub(newGrad, grad, length);
        nextH = next_H(H, DFP_first_term(p, q, length), DFP_sec_term(H, q, length), length);
        H = nextH;
        grad = newGrad;
        it_max++;
    }

    return point;
}

int main () {
    std::vector<double> teste, direction, resultado_teste, vec_gradiente;
    std::vector<std::vector<double>> inversa;
    double step, resultado;
    teste = {1, 1, 1, 1, 1};

    resultado_teste = gradient_method(xSquare, teste, 0.3, 5);

    resultado = xSquare(resultado_teste, 5);
    vec_gradiente = gradient(xSquare, resultado_teste, 5, 0.01);

    for(int i = 0; i < 5; i++) {
        std::cout << vec_gradiente[i] << '\n';
    }

    std::cout << resultado << '\n';

    return 0;
}