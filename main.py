# Загрузим необходимые библиотеки
import numpy as np
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')
# Запишем данные в массивы
a = np.array([60, 70, 20])
b = np.array([40, 30, 30, 50])

c = np.array([[2, 4, 5, 1],
              [2, 3, 9, 4],
              [3, 4, 22, 5]])


def sev_zap(a_, b_, c_):
    a = np.copy(a_)
    b = np.copy(b_)
    c = np.copy(c_)

    # Проверяем условие замкнутости:
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)
    i = 0
    j = 0
    funk = 0
    x = np.zeros((m, n), dtype=int)
    while (i < m) and (j < n):  # повторяем цикл до сходимости метода
        x_ij = min(a[i], b[j])  # проверяем минимальность a_i и b_j
        funk += c[i, j] * min(a[i], b[j])  # записываем в итоговую функцию элемент трат
        a[i] -= x_ij  #
        b[j] -= x_ij  # обновляем векторы a и b
        x[i, j] = x_ij  # добавляем элемент x_ij в матрицу x

        if a[i] > b[j]:  # делаем сдвиги при выполнении условий
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            i += 1
            j += 1
    return x, funk  # возращаем матрицу x и целевую функцию


# Для метода потенциалов потребуется матрица дельт
# На вход она получает x - матрицу одного из опорных методов
def delta(a, b, c, x):
    # Проверяем условие замкнутости:
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)

    u = np.zeros(m)
    v = np.zeros(n)

    for i in range(m):
        for j in range(n):
            if x[i, j] != 0:  # если элемент матрицы x не равен 0, расчитываем для данных индексов векторы u и v
                if v[j] != 0:
                    u[i] = c[i, j] - v[j]
                else:
                    v[j] = c[i, j] - u[i]

    delta = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            delta[i, j] = u[i] + v[j] - c[i, j]  # расчитываем элемент дельта матрицы
    return delta


# Функция возвращает матрицу системы ограничений
def prepare(a, b):
    m = len(a)
    n = len(b)
    h = np.diag(np.ones(n))
    v = np.zeros((m, n))
    v[0] = 1
    for i in range(1, m):
        h = np.hstack((h, np.diag(np.ones(n))))
        k = np.zeros((m, n))
        k[i] = 1
        v = np.hstack((v, k))
    return np.vstack((h, v)).astype(int), np.hstack((b, a))


# Метод потенциалов
def potenz(a_, b_, c_):
    a = np.copy(a_)
    b = np.copy(b_)
    c = np.copy(c_)
    # Проверяем условие замкнутости:
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)
    A_eq, b_eq = prepare(a, b)
    res = linprog(c.reshape(1, -1), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='simplex')
    return res.x.astype(int).reshape(m, n), res.fun.astype(int)  # возращаем матрицу x и целевую функцию


x1, funk1 = sev_zap(a, b, c)
print('\nМетод северо-западного угла: \n', x1)
print('Целевая функция: ', funk1)
print()
print('Дельта матрица для метода северо-западного угла: \n', delta(a, b, c, x1))

x2, funk2 = potenz(a, b, c)
print('\nМетод потенциалов: \n', x2)
print('Целевая функция: ', funk2)
print()
print('Дельта матрица для метода потенциалов: \n', delta(a, b, c, x2))