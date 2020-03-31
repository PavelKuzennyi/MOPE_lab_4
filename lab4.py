import numpy as np
from math import *
from prettytable import PrettyTable
from scipy.stats import f, t
from functools import partial

# Variant №208

while True:
    fisher_t = partial(f.ppf, q=1 - 0.05)
    student_t = partial(t.ppf, q=1 - 0.025)

    x1_min = -5
    x1_max = 15
    x2_min = -35
    x2_max = 10
    x3_min = -35
    x3_max = -10

    x_max_av = (x1_max + x2_max + x3_max) / 3
    x_min_av = (x1_min + x2_min + x3_min) / 3

    y_max = int(200 + x_max_av)
    y_min = int(200 + x_min_av)

    m = 3
    N = 8

    x0_factor = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x1_factor = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    x2_factor = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    x3_factor = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    x1x2_factor = x1_factor * x2_factor
    x1x3_factor = x1_factor * x3_factor
    x2x3_factor = x2_factor * x3_factor
    x1x2x3_factors = x1_factor * x2_factor * x3_factor

    factor_matrix = np.zeros((N, N))

    factor_matrix[0, :] = x0_factor
    factor_matrix[1, :] = x1_factor
    factor_matrix[2, :] = x2_factor
    factor_matrix[3, :] = x3_factor
    factor_matrix[4, :] = x1x2_factor
    factor_matrix[5, :] = x1x3_factor
    factor_matrix[6, :] = x2x3_factor
    factor_matrix[7, :] = x1x2x3_factors

    # print(factor_matrix)

    matrix_plan = np.random.randint(y_min, y_max, size=(N, m))

    y_average = np.zeros((N, 1))
    for i in range(N):
        y_average[i, 0] = round((sum(matrix_plan[i, :] / 3)), 3)

    list_bi = np.zeros((N, 1))
    for i in range(N):
        list_bi[i, 0] = sum(factor_matrix[i, :] * y_average[:, 0] / 3)

    x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x1 = np.array([-5, -5, -5, -5, 15, 15, 15, 15])
    x2 = np.array([-35, -35, 10, 10, -35, -35, 10, 10])
    x3 = np.array([-35, -10, -35, -10, -35, -10, -35, -10])
    x1x2 = x1 * x2
    x1x3 = x1 * x3
    x2x3 = x2 * x3
    x1x2x3 = x1 * x2 * x3

    x_matrix = np.zeros((N, N))

    x_matrix[:, 0] = x0
    x_matrix[:, 1] = x1
    x_matrix[:, 2] = x2
    x_matrix[:, 3] = x3
    x_matrix[:, 4] = x1x2
    x_matrix[:, 5] = x1x3
    x_matrix[:, 6] = x2x3
    x_matrix[:, 7] = x1x2x3

    d_list = np.zeros((N, 1))
    np.array(d_list)
    for i in range(N):
        d_list[i][0] = (round(((matrix_plan[i][0] - y_average[i][0]) ** 2 + (matrix_plan[i][1] - y_average[i][0]) ** 2 + (
                matrix_plan[i][2] - y_average[i][0]) ** 2) / 3, 3))

    d_sum = sum(d_list)

    my_table = np.hstack((x_matrix, matrix_plan, y_average, d_list))

    table = PrettyTable()
    table.field_names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y", "S^2"]
    for i in range(len(my_table)):
        table.add_row(my_table[i])

    print(table)
    print("\ny = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(round(float(list_bi[0]), 3),
                                                                                                 round(float(list_bi[1]), 3),
                                                                                                 round(float(list_bi[2]), 3),
                                                                                                 round(float(list_bi[3]), 3),
                                                                                                 round(float(list_bi[4]), 3),
                                                                                                 round(float(list_bi[5]), 3),
                                                                                                 round(float(list_bi[6]), 3),
                                                                                                 round(float(list_bi[7]), 3)))

    Gp = max(d_list) / d_sum
    F1 = m - 1
    F2 = N
    q = 0.05
    q1 = q / F1
    fisher_value = f.ppf(q=1 - q1, dfn=F2, dfd=(F1 - 1) * F2)
    Gt = fisher_value / (fisher_value + F1 - 1)
    print("Gp = ", float(Gp), "\nGt = ", Gt)

    if Gp < Gt:
        print("Gp < Gt")
        print("Дисперсія однорідна\n")
        dispersion_b = (d_sum / N) / (m * N)
        s_beta = sqrt(abs(dispersion_b))

        beta_list = np.zeros((N, 1))
        for i in range(N):
            beta_list[i, 0] = sum(factor_matrix[i, :] * y_average[:, 0] / N)

        t_list = []
        for i in range(N):
            t_list.append(abs(beta_list[i, 0]) / s_beta)

        F3 = F1 * F2
        d = 0
        T = student_t(df=F3)
        print("t табличне = ", T)
        for i in range(len(t_list)):
            if t_list[i] > T:
                beta_list[i, 0] = 0
                print("Гіпотеза підтверджена, beta{} = 0".format(i))
            else:
                print("Гіпотеза не підтверджена beta{} = {}".format(i, beta_list[i]))
                d += 1

        y_for_student = np.zeros((N, 1))
        for i in range(N):
            y_for_student[i, 0] = sum(x_matrix[i, :] * beta_list[:, 0])

        F4 = N - d
        dispersion = sum(((y_for_student[:][0] - y_average[:][0]) ** 2) * m / (N - d))
        Fp = dispersion / dispersion_b
        Ft = fisher_t(dfn=F4, dfd=F3)
        if Ft > Fp:
            print("Отримана математична модель адекватна експериментальним даним")
            break
        else:
            print("Рівняння регресії неадекватно оригіналу")
            break

    else:
        print("Gp > Gt")
        print("Дисперсія неоднорідна. Спробуйте ще раз.")
        m += 1
