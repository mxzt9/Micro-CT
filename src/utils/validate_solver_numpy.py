"""Валидация логики PNM-решателя на numpy (torch недоступен в песочнице).
Повторяет ту же сборку лапласиана и решение, что и torch-версия."""
import numpy as np


def build_laplacian(g, edge_index, n):
    L = np.zeros((n, n))
    i, j = edge_index
    np.add.at(L, (i, j), -g)
    np.add.at(L, (j, i), -g)
    np.add.at(L, (i, i), g)
    np.add.at(L, (j, j), g)
    return L


def solve_axis(g, edge_index, coords, axis, domain_length, cross_area, mu=1e-3, frac=0.05, eps=1e-12):
    n = coords.shape[0]
    ca = coords[:, axis]
    lo = ca.min() + frac * (ca.max() - ca.min())
    hi = ca.max() - frac * (ca.max() - ca.min())
    inlet = ca <= lo
    outlet = ca >= hi
    fixed = inlet | outlet
    free = ~fixed
    L = build_laplacian(g, edge_index, n)
    P = np.zeros(n)
    P[inlet] = 1.0
    idx_free = np.where(free)[0]
    idx_fixed = np.where(fixed)[0]
    L_ff = L[np.ix_(idx_free, idx_free)] + eps * np.eye(len(idx_free))
    rhs = -L[np.ix_(idx_free, idx_fixed)] @ P[idx_fixed]
    P[idx_free] = np.linalg.solve(L_ff, rhs)
    i_e, j_e = edge_index
    flow_e = g * (P[i_e] - P[j_e])
    Q = (flow_e * inlet[i_e]).sum() - (flow_e * inlet[j_e]).sum()
    k = abs(Q * mu * domain_length / (cross_area * 1.0 + eps))
    return k, Q, inlet.sum(), outlet.sum()


print("--- Тест 1: цепочка из M узлов (резисторы последовательно) ---")
M = 11
g_val = 2.0
coords = np.zeros((M, 3))
coords[:, 0] = np.arange(M)             # вдоль оси z
edge_index = np.array([[i for i in range(M - 1)], [i + 1 for i in range(M - 1)]])
g = np.full(M - 1, g_val)
k, Q, n_in, n_out = solve_axis(g, edge_index, coords, axis=0, domain_length=1.0, cross_area=1.0)
G_eff_analytic = g_val / (M - 1)        # последовательное соединение
print(f"inlet={n_in}, outlet={n_out}")
print(f"Q (solver)        = {Q:.8f}")
print(f"G_eff аналитика   = {G_eff_analytic:.8f}")
print(f"совпадение       : {np.isclose(Q, G_eff_analytic)}")

print("\n--- Тест 2: 2 параллельных цепочки (проводимости складываются) ---")
# две независимые цепочки по M узлов -> G = 2 * g/(M-1)
M2 = 6
coords2 = np.zeros((2 * M2, 3))
coords2[:M2, 0] = np.arange(M2)
coords2[M2:, 0] = np.arange(M2)
coords2[M2:, 1] = 1.0
e1 = [[i for i in range(M2 - 1)], [i + 1 for i in range(M2 - 1)]]
e2 = [[M2 + i for i in range(M2 - 1)], [M2 + i + 1 for i in range(M2 - 1)]]
edge_index2 = np.array([e1[0] + e2[0], e1[1] + e2[1]])
g2 = np.full(edge_index2.shape[1], g_val)
_, Q2, n_in2, n_out2 = solve_axis(g2, edge_index2, coords2, axis=0, domain_length=1.0, cross_area=1.0)
G_eff2 = 2 * g_val / (M2 - 1)
print(f"inlet={n_in2}, outlet={n_out2}")
print(f"Q (solver)      = {Q2:.8f}")
print(f"G_eff аналитика = {G_eff2:.8f}")
print(f"совпадение     : {np.isclose(Q2, G_eff2)}")

print("\n--- Тест 3: монотонность k по g ---")
ks = []
for gg in [0.5, 1.0, 2.0, 4.0]:
    g_arr = np.full(M - 1, gg)
    k, _, _, _ = solve_axis(g_arr, edge_index, coords, axis=0, domain_length=1.0, cross_area=1.0)
    ks.append(k)
print("g  :", [0.5, 1.0, 2.0, 4.0])
print("k  :", [f"{x:.4f}" for x in ks])
print("монотонно растёт:", all(ks[i] < ks[i+1] for i in range(len(ks)-1)))
