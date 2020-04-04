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

    m = 3
    N = 4

    x_max_av = (x1_max + x2_max + x3_max) / 3
    x_min_av = (x1_min + x2_min + x3_min) / 3

    y_max = int(200 + x_max_av)
    y_min = int(200 + x_min_av)

    x_norm = np.array([
        [1, -1, -1, -1],
        [1, -1, 1, 1],
        [1, 1, -1, 1],
        [1, 1, 1, -1]])

    matrix_plan_1 = np.random.randint(y_min, y_max, size=(4, m))
    matrix_plan_2 = np.random.randint(y_min, y_max, size=(4, m))
    matrix_plan = np.vstack((matrix_plan_1, matrix_plan_2))

    while True:
        y_average = np.zeros((N, 1))
        for i in range(N):
            y_average[i, 0] = round((sum(matrix_plan_1[i, :] / m)), 3)

        x_matrix = np.array([
            [x1_min, x2_min, x3_min],
            [x1_min, x2_max, x3_max],
            [x1_max, x2_min, x3_max],
            [x1_max, x2_max, x3_min]
        ])

        mx1 = sum(x_matrix[:, 0] / N)
        mx2 = sum(x_matrix[:, 1] / N)
        mx3 = sum(x_matrix[:, 2] / N)
        my = (float(sum(y_average))) / 4

        a1 = (x_matrix[0][0] * y_average[0][0] + x_matrix[1][0] * y_average[1][0] + x_matrix[2][0] * y_average[2][0] +
              x_matrix[3][0] * y_average[3][0]) / 4
        a2 = (x_matrix[0][1] * y_average[0][0] + x_matrix[1][1] * y_average[1][0] + x_matrix[2][1] * y_average[2][0] +
              x_matrix[3][1] * y_average[3][0]) / 4
        a3 = (x_matrix[0][2] * y_average[0][0] + x_matrix[1][2] * y_average[1][0] + x_matrix[2][2] * y_average[2][0] +
              x_matrix[3][2] * y_average[3][0]) / 4
        a11 = (x_matrix[0][0] ** 2 + x_matrix[1][0] ** 2 + x_matrix[2][0] ** 2 + x_matrix[3][0] ** 2) / 4
        a22 = (x_matrix[0][1] ** 2 + x_matrix[1][1] ** 2 + x_matrix[2][1] ** 2 + x_matrix[3][1] ** 2) / 4
        a33 = (x_matrix[0][2] ** 2 + x_matrix[1][2] ** 2 + x_matrix[2][2] ** 2 + x_matrix[3][2] ** 2) / 4
        a12 = a21 = (x_matrix[0][0] * x_matrix[0][1] + x_matrix[1][0] * x_matrix[1][1] +
                     x_matrix[2][0] * x_matrix[2][1] + x_matrix[3][0] * x_matrix[3][1]) / 4
        a13 = a31 = (x_matrix[0][0] * x_matrix[0][2] + x_matrix[1][0] * x_matrix[1][2] +
                     x_matrix[2][0] * x_matrix[2][2] + x_matrix[3][0] * x_matrix[3][2]) / 4
        a23 = a32 = (x_matrix[0][1] * x_matrix[0][2] + x_matrix[1][1] * x_matrix[1][2] +
                     x_matrix[2][1] * x_matrix[2][2] + x_matrix[3][1] * x_matrix[3][2]) / 4

        znam_matrix = [
            [1, mx1, mx2, mx3],
            [mx1, a11, a12, a13],
            [mx2, a12, a22, a32],
            [mx3, a13, a23, a33]
        ]

        b0_matrix = [
            [my, mx1, mx2, mx3],
            [a1, a11, a12, a13],
            [a2, a12, a22, a32],
            [a3, a13, a23, a33]
        ]

        b1_matrix = [
            [1, my, mx2, mx3],
            [mx1, a1, a12, a13],
            [mx2, a2, a22, a32],
            [mx3, a3, a23, a33]
        ]

        b2_matrix = [
            [1, mx1, my, mx3],
            [mx1, a11, a1, a13],
            [mx2, a12, a2, a32],
            [mx3, a13, a3, a33]
        ]

        b3_matrix = [
            [1, mx1, mx2, my],
            [mx1, a11, a12, a1],
            [mx2, a12, a22, a2],
            [mx3, a13, a23, a3]
        ]

        b0 = np.linalg.det(b0_matrix) / np.linalg.det(znam_matrix)
        b1 = np.linalg.det(b1_matrix) / np.linalg.det(znam_matrix)
        b2 = np.linalg.det(b2_matrix) / np.linalg.det(znam_matrix)
        b3 = np.linalg.det(b3_matrix) / np.linalg.det(znam_matrix)

        table = PrettyTable()
        my_table = np.hstack((x_matrix, matrix_plan_1))
        table.field_names = ["X1", "X2", "X3", "Y1", "Y2", "Y3"]
        for i in range(len(my_table)):
            table.add_row(my_table[i])

        print(table)

        print("\nb0:", "%.3f " % b0, "\nb1:", "%.3f" % b1, "\nb2:", "%.3f" % b2, "\nb3:", "%.3f\n" % b3)
        print(f"Рівняння регресії: y = {b0:.3f}{b1:+.3f}*x1{b2:+.3f}*x2{b3:+.3f}*x3")

        print("b0 + b1*X11 + b2*X12 + b3*X13 =",
              "%.2f" % (b0 + b1 * x_matrix[0][0] + b2 * x_matrix[0][1] + b3 * x_matrix[0][2]),
              "| y1 =", "%.2f" % y_average[0, 0])
        print("b0 + b1*X21 + b2*X22 + b3*X23 =",
              "%.2f" % (b0 + b1 * x_matrix[1][0] + b2 * x_matrix[1][1] + b3 * x_matrix[1][2]),
              "| y2 =", "%.2f" % y_average[1, 0])
        print("b0 + b1*X31 + b2*X32 + b3*X33 =",
              "%.2f" % (b0 + b1 * x_matrix[2][0] + b2 * x_matrix[2][1] + b3 * x_matrix[2][2]),
              "| y3 =", "%.2f" % y_average[2, 0])
        print("b0 + b1*X41 + b2*X42 + b3*X43 =",
              "%.2f" % (b0 + b1 * x_matrix[3][0] + b2 * x_matrix[3][1] + b3 * x_matrix[3][2]),
              "| y4 =", "%.2f" % y_average[3, 0])

        d1 = ((matrix_plan_1[0][0] - y_average[0, 0]) ** 2 + (matrix_plan_1[0][1] - y_average[0, 0]) ** 2 + (
                matrix_plan_1[0][2] - y_average[0, 0]) ** 2) / 3
        d2 = ((matrix_plan_1[1][0] - y_average[1, 0]) ** 2 + (matrix_plan_1[1][1] - y_average[1, 0]) ** 2 + (
                matrix_plan_1[1][2] - y_average[1, 0]) ** 2) / 3
        d3 = ((matrix_plan_1[2][0] - y_average[2, 0]) ** 2 + (matrix_plan_1[2][1] - y_average[2, 0]) ** 2 + (
                matrix_plan_1[2][2] - y_average[2, 0]) ** 2) / 3
        d4 = ((matrix_plan_1[3][0] - y_average[3, 0]) ** 2 + (matrix_plan_1[3][1] - y_average[3, 0]) ** 2 + (
                matrix_plan_1[3][2] - y_average[3, 0]) ** 2) / 3

        d_matrix = [d1, d2, d3, d4]

        Gp = max(d_matrix) / sum(d_matrix)

        m = len(matrix_plan_1[0])
        f1 = m - 1
        f2 = N = len(x_matrix)
        q = 0.05
        Gt = 0.7679

        print("\n-------------------")

        print("\nКритерій Фішера")

        print("\nGp = %.4f" % Gp)
        print("Gt =", Gt, "\n")

        if Gp < Gt:
            print("%.4f < %.4f " % (Gp, Gt))
            print("Дисперсія однорідна\n")
            break
        else:
            print("%.4f > %.4f " % (Gp, Gt))
            print("Дисперсія не однорідна\n")
            m += 1
            rand_string = np.random.randint(y_min, y_max, size=(4, 1))
            matrix_plan_1 = np.hstack(matrix_plan_1, rand_string)

    S2 = sum(d_matrix) / N
    S2b = S2 / (N * m)
    Sb = sqrt(S2b)

    y_list = [y_average[0, 0], y_average[1, 0], y_average[2, 0], y_average[3, 0]]

    B0 = sum(y_list * x_norm[:, 0]) / N
    B1 = sum(y_list * x_norm[:, 1]) / N
    B2 = sum(y_list * x_norm[:, 2]) / N
    B3 = sum(y_list * x_norm[:, 3]) / N

    t0 = fabs(B0) / Sb
    t1 = fabs(B1) / Sb
    t2 = fabs(B2) / Sb
    t3 = fabs(B3) / Sb

    print("-------------------")

    print("\nКритерій Стьюдента\n")

    p = 0.95
    f3 = f1 * f2
    t_tab = t.ppf((1 + p) / 2, f3)
    print("t0:", "%.3f " % t0, "\nt1:", "%.3f" % t1, "\nt2:", "%.3f" % t2, "\nt3:", "%.3f\n" % t3)
    if t0 < t_tab:
        b0 = 0
        print("t0 < t_таб; отже b0=0")

    if t1 < t_tab:
        b1 = 0
        print("t1 < t_таб; отже b1=0")

    if t2 < t_tab:
        b2 = 0
        print("t2 < t_таб; отже b2=0")

    if t3 < t_tab:
        b3 = 0
        print("t3 < t_таб; отже b3=0")

    y1_cov = b0 + b1 * x_matrix[0][0] + b2 * x_matrix[0][1] + b3 * x_matrix[0][2]
    y2_cov = b0 + b1 * x_matrix[1][0] + b2 * x_matrix[1][1] + b3 * x_matrix[1][2]
    y3_cov = b0 + b1 * x_matrix[2][0] + b2 * x_matrix[2][1] + b3 * x_matrix[2][2]
    y4_cov = b0 + b1 * x_matrix[3][0] + b2 * x_matrix[3][1] + b3 * x_matrix[3][2]

    print("\ny1:", "%.3f " % y1_cov, "\ny2:", "%.3f" % y2_cov, "\ny3:", "%.3f" % y3_cov, "\ny4:", "%.3f\n" % y4_cov)

    print("-------------------\n")

    print("Критерій Фішера")

    d = 2
    f4 = N - d

    S2_ad = (m / (N - d)) * (
            (y1_cov - y_average[0, 0]) ** 2 + (y2_cov - y_average[1, 0]) ** 2 + (y3_cov - y_average[2, 0]) ** 2 + (
            y4_cov - y_average[3, 0]) ** 2)
    Fp = S2_ad / S2b
    Ft = f.ppf(p, f4, f3)
    print("\nFt =", Ft)
    print("Fp = %.2f" % Fp)
    if Fp > Ft:
        print("Fp > Ft")
        print("Рівняння регресії не адекватно оригіналу при рівні значимості 0,05\n")
    else:
        print("Fp < Ft")
        print("Рівняння регресії адекватно оригіналу при рівні значимості 0,05")
        break

    ######################################################

    print("-------------------\n")

    print("Перейдемо до регресії з ефектом взаємодії")

    N = 8
    y_average = np.zeros((N, 1))
    for i in range(N):
        y_average[i, 0] = round((sum(matrix_plan[i, :] / m)), 3)

    x0_factor = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x1_factor = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    x2_factor = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    x3_factor = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    x1x2_factor = x1_factor * x2_factor
    x1x3_factor = x1_factor * x3_factor
    x2x3_factor = x2_factor * x3_factor
    x1x2x3_factors = x1_factor * x2_factor * x3_factor

    x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x1 = np.array([-5, -5, -5, -5, 15, 15, 15, 15])
    x2 = np.array([-35, -35, 10, 10, -35, -35, 10, 10])
    x3 = np.array([-35, -10, -35, -10, -35, -10, -35, -10])
    x1x2 = x1 * x2
    x1x3 = x1 * x3
    x2x3 = x2 * x3
    x1x2x3 = x1 * x2 * x3

    factor_matrix = np.zeros((N, N))
    factor_matrix[0, :] = x0_factor
    factor_matrix[1, :] = x1_factor
    factor_matrix[2, :] = x2_factor
    factor_matrix[3, :] = x3_factor
    factor_matrix[4, :] = x1x2_factor
    factor_matrix[5, :] = x1x3_factor
    factor_matrix[6, :] = x2x3_factor
    factor_matrix[7, :] = x1x2x3_factors

    x_matrix = np.zeros((N, N))

    x_matrix[:, 0] = x0
    x_matrix[:, 1] = x1
    x_matrix[:, 2] = x2
    x_matrix[:, 3] = x3
    x_matrix[:, 4] = x1x2
    x_matrix[:, 5] = x1x3
    x_matrix[:, 6] = x2x3
    x_matrix[:, 7] = x1x2x3

    list_bi = np.zeros((N, 1))
    for i in range(N):
        list_bi[i, 0] = sum(factor_matrix[i, :] * y_average[:, 0] / 3)

    d_list = np.zeros((N, 1))
    np.array(d_list)
    for i in range(N):
        d_list[i][0] = (
            round(((matrix_plan[i][0] - y_average[i][0]) ** 2 + (matrix_plan[i][1] - y_average[i][0]) ** 2 + (
                    matrix_plan[i][2] - y_average[i][0]) ** 2) / 3, 3))

    d_sum = sum(d_list)

    my_table = np.hstack((x_matrix, matrix_plan, y_average, d_list))

    table = PrettyTable()
    table.field_names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y", "S^2"]
    for i in range(len(my_table)):
        table.add_row(my_table[i])

    print(table)
    print("\ny = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(
        round(float(list_bi[0]), 3),
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
            print("\nРівняння регресії неадекватно оригіналу")
            break

    else:
        print("Gp > Gt")
        print("Дисперсія неоднорідна. Спробуйте ще раз.")
        m += 1
