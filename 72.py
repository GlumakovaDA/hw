import math
import numpy as np
from typing import List, Tuple
import random


class Vector:
    def __init__(self, data: List[float]):
        self.data = data
        self.len = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        return self.len

    def __str__(self):
        return str(self.data)

    def norm_1(self):
        return sum(abs(x) for x in self.data)

    def norm_2(self):
        return math.sqrt(sum(x * x for x in self.data))

    def norm_inf(self) -> float:
        return max(abs(x) for x in self.data)

    def dot(self, other: 'Vector'):
        if self.len != other.len:
            raise ValueError("Векторы разной размерности")
        return sum(self[i] * other[i] for i in range(self.len))

    def __add__(self, other: 'Vector'):
        if self.len != other.len:
            raise ValueError("Векторы разной размерности")
        return Vector([self[i] + other[i] for i in range(self.len)])

    def __sub__(self, other: 'Vector'):
        if self.len != other.len:
            raise ValueError("Векторы разной размерности")
        return Vector([self[i] - other[i] for i in range(self.len)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def __rmul__(self, scalar):
        return self * scalar


class Matrix:
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

        for row in data:
            if len(row) != self.cols:
                raise ValueError("Строки разной длины")

    def __getitem__(self, index):
        i, j = index
        return self.data[i][j]

    def __setitem__(self, index, value):
        i, j = index
        self.data[i][j] = value

    def __str__(self):
        return '\n'.join([' '.join(f'{x:8.4f}' for x in row) for row in self.data])

    def get_row(self, i):
        return Vector(self.data[i][:])

    def get_col(self, j):
        return Vector([self[i, j] for i in range(self.rows)])

    def norm_1(self):
        col_sums = [sum(abs(self[i, j]) for i in range(self.rows)) for j in range(self.cols)]
        return max(col_sums)

    def norm_2(self):
        if self.rows == 0 or self.cols == 0:
            return 0.0
        np_matrix = np.array(self.data)
        sv = np.linalg.svd(np_matrix, compute_uv=False, full_matrices=False)
        return np.max(sv)

    def norm_inf(self):
        row_sums = [sum(abs(self[i, j]) for j in range(self.cols)) for i in range(self.rows)]
        return max(row_sums)

    def norm_frobenius(self) -> float:
        return math.sqrt(sum(self[i, j] ** 2 for i in range(self.rows) for j in range(self.cols)))

    def transpose(self):
        return Matrix([[self[i, j] for i in range(self.rows)] for j in range(self.cols)])

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Разные размеры")

        result = [[self[i, j] + other[i, j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Разные размеры")

        result = [[self[i, j] - other[i, j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def __matmul__(self, other):
        if isinstance(other, Vector):
            if self.cols != other.len:
                raise ValueError("Несовместимые размеры")

            result = [0.0] * self.rows
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i] += self[i, j] * other[j]
            return Vector(result)

        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Несовместимые размеры")

            result = [[0.0] * other.cols for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result[i][j] += self[i, k] * other[k, j]
            return Matrix(result)

    def input_matrix():
        print("Введите матрицу построчно. Числа в строке разделяйте пробелами.")
        matrix_data = []
        try:
            while True:
                line = input().strip()
                if line == "":
                    if not matrix_data:
                        print("Матрица не введена!")
                        return None
                    break
                row = list(map(float, line.split()))
                matrix_data.append(row)
            cols = len(matrix_data[0])
            for i, row in enumerate(matrix_data):
                if len(row) != cols:
                    print(f"Ошибка: строка {i+1} имеет разную длину!")
                    return None
            return Matrix(matrix_data)
        except ValueError:
            print("Ошибка: вводите только числа!")
            return None

        def condition_number(matr, norm_type):
            try:
                A_inv = matr.inverse()

                if norm_type == '1':
                    norm_A = matr.norm_1()
                    norm_A_inv = A_inv.norm_1()
                elif norm_type == '2':
                    norm_A = matr.norm_2()
                    norm_A_inv = A_inv.norm_2()
                elif norm_type == 'inf':
                    norm_A = matr.norm_inf()
                    norm_A_inv = A_inv.norm_inf()
                elif norm_type == 'fro':
                    norm_A = matr.norm_frobenius()
                    norm_A_inv = A_inv.norm_frobenius()
                else:
                    raise ValueError("Неизвестный тип нормы. Используйте '1', '2', 'inf' или 'fro'")

                return norm_A * norm_A_inv

            except ValueError as e:
                print(f"Ошибка при вычислении числа обусловленности: {e}")
                return float('inf')


def analyze(A, b: Vector = None):
    norms = ['1', '2', 'inf', 'fro']
    for norm in norms:
        cond = A.condition_number(norm)
        print(f"\nЧисло обусловленности (норма {norm}): {cond:.6e}")

        if cond > 1e15:
            print(f"Плохо обусловлена")
        else:
            print(f"Хорошо обусловлена")


def gauss_solve(A, b, pivot_type: str = 'none'):
    n = A.rows
    mat = [[A[i, j] for j in range(n)] for i in range(n)]
    vec = [b[i] for i in range(n)]
    operations_count = 0

    col_perm = list(range(n))

    for k in range(n):
        if pivot_type == 'column':
            max_row = max(range(k, n), key=lambda i: abs(mat[i][k]))
            if max_row != k:
                mat[k], mat[max_row] = mat[max_row], mat[k]
                vec[k], vec[max_row] = vec[max_row], vec[k]

        elif pivot_type == 'row':
            max_col = max(range(k, n), key=lambda j: abs(mat[k][j]))
            if max_col != k:
                for i in range(n):
                    mat[i][k], mat[i][max_col] = mat[i][max_col], mat[i][k]
                col_perm[k], col_perm[max_col] = col_perm[max_col], col_perm[k]

        elif pivot_type == 'full':
            max_val, max_i, max_j = 0, k, k
            for i in range(k, n):
                for j in range(k, n):
                    if abs(mat[i][j]) > abs(max_val):
                        max_val = mat[i][j]
                        max_i, max_j = i, j

            if max_i != k:
                mat[k], mat[max_i] = mat[max_i], mat[k]
                vec[k], vec[max_i] = vec[max_i], vec[k]

            if max_j != k:
                for i in range(n):
                    mat[i][k], mat[i][max_j] = mat[i][max_j], mat[i][k]
                col_perm[k], col_perm[max_j] = col_perm[max_j], col_perm[k]

        pivot = mat[k][k]

        for i in range(k + 1, n):
            factor = mat[i][k] / pivot
            operations_count += 1

            for j in range(k, n):
                mat[i][j] -= factor * mat[k][j]
                operations_count += 2

            vec[i] -= factor * vec[k]
            operations_count += 2

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = vec[i]
        for j in range(i + 1, n):
            s -= mat[i][j] * x[j]
            operations_count += 2
        x[i] = s / mat[i][i]
        operations_count += 1

    if pivot_type == 'full' or pivot_type == 'row':
        x_temp = x[:]
        for i in range(n):
            x[col_perm[i]] = x_temp[i]

    return Vector(x), operations_count




def generate_matrix(n, min_val=-10, max_val=10, ensure_invertible=True):
    A = np.random.uniform(min_val, max_val, size=(n, n))
    return A


def lup_decomposition(A):
    n = A.rows
    L = [[0.0] * n for _ in range(n)]
    U = [[A[i, j] for j in range(n)] for i in range(n)]
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    op = 0

    for k in range(n):
        max_index = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_index = i

        if max_index != k:
            U[k], U[max_index] = U[max_index], U[k]
            P[k], P[max_index] = P[max_index], P[k]
            if k > 0:
                for j in range(k):
                    L[k][j], L[max_index][j] = L[max_index][j], L[k][j]

        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            op += 1
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * U[k][j]
                op += 2
            U[i][k] = 0.0

    return Matrix(L), Matrix(U), Matrix(P), op


def lu_decomposition(A):
    n = A.rows
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    op = 0

    for i in range(n):
        for j in range(i, n):
            summ = 0.0
            for k in range(i):
                summ += L[i][k] * U[k][j]
                op += 2
            U[i][j] = A[i, j] - summ
            op += 1

        L[i][i] = 1.0
        for j in range(i + 1, n):
            summ = 0.0
            for k in range(i):
                summ += L[j][k] * U[k][i]
                op += 2
            L[j][i] = (A[j, i] - summ) / U[i][i]
            op += 2

    return Matrix(L), Matrix(U), op


def lu_solve(L, U, b):
    n = L.rows
    ops = 0
    y = [0.0] * n
    for i in range(n):
        summ = b[i]
        for j in range(i):
            summ -= L[i, j] * y[j]
            ops += 2
        y[i] = summ

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        summ = y[i]
        for j in range(i + 1, n):
            summ -= U[i, j] * x[j]
            ops += 2
        x[i] = summ / U[i, i]
        ops += 1

    return Vector(x), ops


def lup_solve(L, U, P, b):
    ops = 0
    n = P.rows
    Pb_data = [0.0] * len(b)
    for i in range(n):
        for j in range(n):
            Pb_data[i] += P[i, j] * b[j]
            ops+=2
    Pb = Vector(Pb_data)

    x, ops_lu = lu_solve(L, U, Pb)
    ops += ops_lu

    return x, ops


def determinant_from_lu(U):
    det = 1.0
    for i in range(U.rows):
        det *= U[i, i]
    return det


def max_abs_error(A, x, b):
    Ax = A @ x
    return max(abs(Ax[i]-b[i]) for i in range(len(b)))

def run_all_tests(n, A_np, x_true_data):
    A = Matrix(A_np.tolist())
    x_true = Vector(x_true_data.tolist())
    b = A @ x_true

    print("\nИсходные данные:")
    print("Матрица A:\n", A)
    print("Вектор b:", b)
    print("Точное решение x_exact:", x_true)

    print("\n========== МЕТОД ГАУССА ==========")
    x_g, cnt_g = gauss_solve(A, b)
    diff_g = Vector([x_g[i] - x_true[i] for i in range(n)]).norm_2()
    print("x_calculated =", x_g)
    print("diff =", diff_g)
    print("operations =", cnt_g)

    print("\n========== МЕТОД LU ==========")
    L, U, ops_lu = lu_decomposition(A)
    print("Матрица L:\n", L)
    print("Матрица U:\n", U)
    det_A = determinant_from_lu(U)
    print("det(A) =", det_A)
    x_lu, cnt_lu = lu_solve(L, U, b)
    diff_lu = Vector([x_lu[i] - x_true[i] for i in range(n)]).norm_2()
    print("x_calculated =", x_lu)
    print("diff =", diff_lu)
    print("operations =", cnt_lu + ops_lu)

    print("\n========== МЕТОД LUP ==========")
    L, U, P, ops_lup = lup_decomposition(A)
    print("Матрицы L, U, P:")
    print("L =\n", L)
    print("U =\n", U)
    print("P =\n", P)
    x_lup, cnt_lup = lup_solve(L, U, P, b)
    diff_lup = Vector([x_lup[i] - x_true[i] for i in range(n)]).norm_2()
    print("x_calculated =", x_lup)
    print("diff =", diff_lup)
    print("operations =", cnt_lup + ops_lup)

    print(f"\nСравнение по операциям:")
    print(f"Gauss: {cnt_g}")
    print(f"LU: {cnt_lu + ops_lu}")
    print(f"LUP: {cnt_lup + ops_lup}")

n = 3
A_np = generate_matrix(n)
x_true_data = np.random.uniform(-10, 10, n)
run_all_tests(n, A_np, x_true_data)

