import numpy as np
import math
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

    def inverse(matr):
        if matr.rows != matr.cols:
            raise ValueError("Матрица должна быть квадратной")

        n = matr.rows
        augmented = []
        for i in range(n):
            row = matr.data[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
            augmented.append(row)

        for col in range(n):
            pivot_row = col
            for i in range(col + 1, n):
                if abs(augmented[i][col]) > abs(augmented[pivot_row][col]):
                    pivot_row = i

            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

            pivot_val = augmented[col][col]
            if abs(pivot_val) < 1e-10:
                raise ValueError("Матрица вырождена или плохо обусловлена")

            for j in range(2 * n):
                augmented[col][j] /= pivot_val

            for i in range(n):
                if i != col:
                    factor = augmented[i][col]
                    for j in range(2 * n):
                        augmented[i][j] -= factor * augmented[col][j]

        inverse_data = [row[n:2 * n] for row in augmented]
        return Matrix(inverse_data)

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

def gauss_solve(A: Matrix, b: Vector, pivot_type: str = 'none'):
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
                operations_count += n + 1

        elif pivot_type == 'row':
            max_col = max(range(k, n), key=lambda j: abs(mat[k][j]))
            if max_col != k:
                for i in range(n):
                    mat[i][k], mat[i][max_col] = mat[i][max_col], mat[i][k]
                operations_count += n
                col_perm[k], col_perm[max_col] = col_perm[max_col], col_perm[k]

        elif pivot_type == 'full':
            max_val, max_i, max_j = 0, k, k
            for i in range(k, n):
                for j in range(k, n):
                    if abs(mat[i][j]) > abs(max_val):
                        max_val = mat[i][j]
                        max_i, max_j = i, j
            operations_count += (n - k) * (n - k)

            if max_i != k:
                mat[k], mat[max_i] = mat[max_i], mat[k]
                vec[k], vec[max_i] = vec[max_i], vec[k]
                operations_count += n + 1

            if max_j != k:
                for i in range(n):
                    mat[i][k], mat[i][max_j] = mat[i][max_j], mat[i][k]
                operations_count += n
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


def cholesky_decomposition(A: Matrix):

    n = A.rows
    L = [[0.0] * n for _ in range(n)]
    op = 0

    for i in range(n):
        sum_sq = 0.0
        for k in range(i):
            sum_sq += L[i][k] ** 2
            op += 2

        diag_el = A[i, i] - sum_sq
        if diag_el <= 0:
            raise ValueError(f"Матрица не является положительно определенной")

        L[i][i] = math.sqrt(diag_el)
        op += 2

        for j in range(i + 1, n):
            sum_prod = 0.0
            for k in range(i):
                sum_prod += L[j][k] * L[i][k]
                op += 2

            L[j][i] = (A[j, i] - sum_prod) / L[i][i]
            op += 2

    return Matrix(L), op


def cholesky_solve(L, b):
    n = len(b)
    y = Vector([0.0] * n)
    x = Vector([0.0] * n)
    ops = 0

    for i in range(n):
        sum_ly = 0.0
        for j in range(i):
            sum_ly += L[i, j] * y[j]
            ops += 2
        y[i] = (b[i] - sum_ly) / L[i, i]
        ops += 2

    for i in range(n-1, -1, -1):
        sum_ltx = 0.0
        for j in range(i+1, n):
            sum_ltx += L[j, i] * x[j]
            ops += 2
        x[i] = (y[i] - sum_ltx) / L[i, i]
        ops += 2

    return x, ops



def generate_matrix(n):
    A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)], dtype=float)
    for i in range(n):
        A[i, i] += 1e-7
    return A


def generate_spd_matrix(n):
    A = np.random.uniform(-1, 1, (n, n))

    A = (A + A.T) / 2
    A = A.T @ A + n * np.eye(n)

    return A


def generate_non_spd_matrix(n: int):
    A = np.random.uniform(-10, 10, (n, n))
    for i in range(n):
        A[i, i] = -abs(A[i, i])
    return Matrix(A.tolist())


def generate_indefinite_matrix(n: int):
    A = np.random.uniform(-1, 1, (n, n))
    A = (A + A.T) / 2
    A[0, 0] = -abs(A[0, 0])
    return Matrix(A.tolist())


def is_symmetric(A: Matrix, tol=1e-10):
    if A.rows != A.cols:
        return False
    for i in range(A.rows):
        for j in range(i + 1, A.cols):
            if abs(A[i, j] - A[j, i]) > tol:
                return False
    return True


def run_all_tests(n, A, x_true_data):
    if hasattr(A, 'shape'):
        A = Matrix(A.tolist())
    if hasattr(x_true_data, 'tolist'):
        x_true = Vector(x_true_data.tolist())
    else:
        x_true = Vector(x_true_data)

    b = A @ x_true

    print("\n========== МЕТОД ГАУССА ==========")
    print("\nМатрица A:\n", A)
    x_g, cnt_g = gauss_solve(A, b)
    diff_g = Vector([x_g[i] - x_true[i] for i in range(n)]).norm_2()

    print("x_calculated =", x_g)
    print("x_exact =", x_true)
    print("diff =", diff_g)
    print("operations =", cnt_g)

    print("\n========== МЕТОД LU ==========")
    L, U, ops_lu = lu_decomposition(A)
    print("\nМатрица L:\n", L)
    print("\nМатрица U:\n", U)
    x_lu, cnt_lu = lu_solve(L, U, b)
    diff_lu = Vector([x_lu[i] - x_true[i] for i in range(n)]).norm_2()

    print("diff =", diff_lu)
    print("operations =", cnt_lu + ops_lu)

    print("\n========== МЕТОД LUP ==========")
    L, U, P, ops_lup = lup_decomposition(A)
    print("\nМатрицы L, U, P:")
    print("\nL =\n", L)
    print("\nU =\n", U)
    print("\nP =\n", P)
    x_lup, cnt_lup = lup_solve(L, U, P, b)
    diff_lup = Vector([x_lup[i] - x_true[i] for i in range(n)]).norm_2()


    print("diff =", diff_lup)
    print("operations =", cnt_lup + ops_lup)

    print("\n========== МЕТОД ХОЛЕЦКОГО ==========")
    try:
        L_chol, ops_chol_decomp = cholesky_decomposition(A)
        print("\nМатрица L (разложение Холецкого):\n", L_chol)

        L_LT = L_chol @ L_chol.transpose()
        reconstruction_error = max(abs(L_LT[i, j] - A[i, j]) for i in range(n) for j in range(n))
        print("Ошибка восстановления A = L*L^T =", reconstruction_error)

        x_chol, ops_chol_solve = cholesky_solve(L_chol, b)
        diff_chol = Vector([x_chol[i] - x_true[i] for i in range(n)]).norm_2()


        print("x_calculated =", x_chol)
        print("x_exact =", x_true)
        print("diff =", diff_chol)
        print("operations =", ops_chol_decomp + ops_chol_solve)

    except ValueError as e:
        print("Метод Холецкого не применим:", e)
        ops_chol_decomp = 0
        ops_chol_solve = 0

    print(f"\nСравнение по операциям:")
    print(f"Gauss: {cnt_g}")
    print(f"LU: {cnt_lu + ops_lu}")
    print(f"LUP: {cnt_lup + ops_lup}")
    print(f"Cholesky: {ops_chol_decomp + ops_chol_solve}")


def test_cholesky():
    n = 5

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МЕТОДА ХОЛЕЦКОГО НА РАЗНЫХ ТИПАХ МАТРИЦ")
    print("=" * 60)

    print("\n1. SPD МАТРИЦА :")
    A_spd_np = generate_spd_matrix(n)
    if hasattr(A_spd_np, 'tolist'):
        A_spd = Matrix(A_spd_np.tolist())
    else:
        A_spd = A_spd_np
    x_true = Vector([random.uniform(-10, 10) for _ in range(n)])
    b = A_spd @ x_true

    print("Матрица A:\n", A_spd)
    print("Симметрична:", is_symmetric(A_spd))
    try:
        L, ops_decomp = cholesky_decomposition(A_spd)
        x, ops_solve = cholesky_solve(L, b)
        error = Vector([x[i] - x_true[i] for i in range(n)]).norm_2()
        print("УСПЕХ")
        print(f"  Ошибка: {error:.2e}")
        print(f"  Операции: {ops_decomp + ops_solve}")
    except ValueError as e:
        print("ОШИБКА", e)

    print("\n2. НЕСИММЕТРИЧНАЯ МАТРИЦА :")
    A_non_sym = generate_non_spd_matrix(n)
    print("Матрица A:\n", A_non_sym)
    print("Симметрична:", is_symmetric(A_non_sym))
    try:
        L, ops_decomp = cholesky_decomposition(A_non_sym)
        print("УСПЕХ")
    except ValueError as e:
        print("ОШИБКА ", e)

    print("\n3. СИММЕТРИЧНАЯ НЕ-ПО МАТРИЦА :")
    A_indef = generate_indefinite_matrix(n)
    print("Матрица A:\n", A_indef)
    print("Симметрична:", is_symmetric(A_indef))
    try:
        L, ops_decomp = cholesky_decomposition(A_indef)
        print("УСПЕХ")
    except ValueError as e:
        print("ОШИБКА", e)


def compare_cholesky_theory():
    n = 5
    A_spd_np = generate_spd_matrix(n)
    A_spd = Matrix(A_spd_np.tolist())

    x_true = Vector([random.uniform(-10, 10) for _ in range(n)])
    b = A_spd @ x_true

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С ТЕОРЕТИЧЕСКИМИ ОЦЕНКАМИ")
    print("=" * 60)

    L, chol_decomp_ops = cholesky_decomposition(A_spd)
    x_chol, chol_solve_ops = cholesky_solve(L, b)
    chol_total = chol_decomp_ops + chol_solve_ops

    x_gauss, gauss_ops = gauss_solve(A_spd, b, pivot_type='none')

    print(f"\nПрактические результаты (n={n}):")
    print(f"Холецкий: {chol_total} операций")
    print(f"Гаусс: {gauss_ops} операций")

    print(f"\nТеоретические оценки (n={n}):")
    chol_theory = (n ** 3) / 3 + 2 * (n ** 2)
    gauss_theory = 2 * (n ** 3) / 3
    print(f"Холецкий: ~{chol_theory:.0f} операций")
    print(f"Гаусс: ~{gauss_theory:.0f} операций")


if __name__ == "__main__":
    n = 5
    A_np = generate_spd_matrix(n)
    x_true_data = np.random.uniform(-10, 10, n)
    run_all_tests(n, A_np, x_true_data)

    test_cholesky()
    compare_cholesky_theory()