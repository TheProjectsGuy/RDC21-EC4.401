# %% Import everything
import sympy as sp
import numpy as np
from IPython.display import display

# %% Variables
# Joint angles
t1, t2, t3 = sp.symbols(r"\theta_1, \theta_2, \theta_3")
# Link lengths
l1, l2, l3 = sp.symbols(r"l_1, l_2, l_3")
# C.O.M. offsets
r1, r2, r3 = sp.symbols(r"r_1, r_2, r_3")
# Link masses
m1, m2, m3 = sp.symbols(r"m_1, m_2, m_3")
# Moment of inertias (all along Z axis only)
izz1, izz2, izz3 = sp.symbols(r"I_{zz_1}, I_{zz_2}, I_{zz_3}")
Icc1, Icc2, Icc3 = [sp.Matrix(np.diag([0, 0, izz])) for izz in \
    [izz1, izz2, izz3]]

# %% Forward kinematics
pr1 = sp.Matrix([
    [r1*sp.cos(t1)],
    [r1*sp.sin(t1)],
    [0],
    [0], [0], [t1]
])  # Pose of COM (x, y, z, tx, ty, tz) of link 1 -> r1
pr2 = sp.Matrix([
    [l1*sp.cos(t1) + r2*sp.cos(t1+t2)],
    [l1*sp.sin(t1) + r2*sp.sin(t1+t2)],
    [0],
    [0], [0], [t1+t2]
])  # Pose of COM of link 2 -> r2
pr3 = sp.Matrix([
    [l1*sp.cos(t1) + l2*sp.cos(t1+t2) + r3*sp.cos(t1+t2+t3)],
    [l1*sp.sin(t1) + l2*sp.sin(t1+t2) + r3*sp.sin(t1+t2+t3)],
    [0],
    [0], [0], [t1+t2+t3]
])  # Pose of COM of link 3 -> r3
pef = sp.Matrix([
    [l1*sp.cos(t1) + l2*sp.cos(t1+t2) + l3*sp.cos(t1+t2+t3)],
    [l1*sp.sin(t1) + l2*sp.sin(t1+t2) + l3*sp.sin(t1+t2+t3)],
    [0],
    [0], [0], [t1+t2+t3]
])  # Pose of end effector -> pef

# %% Shorthand subs
sh_subs = {
    sp.sin(t1): sp.symbols(r"s_1"),
    sp.cos(t1): sp.symbols(r"c_1"),
    sp.sin(t2): sp.symbols(r"s_2"),
    sp.cos(t2): sp.symbols(r"c_2"),
    sp.sin(t3): sp.symbols(r"s_3"),
    sp.cos(t3): sp.symbols(r"c_3"),
    sp.sin(t1+t2): sp.symbols(r"s_{12}"),
    sp.cos(t1+t2): sp.symbols(r"c_{12}"),
    sp.sin(t2+t3): sp.symbols(r"s_{23}"),
    sp.cos(t2+t3): sp.symbols(r"c_{23}"),
    sp.sin(t1+t2+t3): sp.symbols(r"s_{123}"),
    sp.cos(t1+t2+t3): sp.symbols(r"c_{123}")
}   # Angles in short hand

# %% Jacobians
Jv1 = sp.Matrix.hstack(pr1.diff(t1), pr1.diff(t2), 
    pr1.diff(t3))[0:3,:]    # Jv1: Velocity (Pr1)
Jv2 = sp.Matrix.hstack(pr2.diff(t1), pr2.diff(t2), 
    pr2.diff(t3))[0:3,:]    # Jv2: Velocity (Pr2)
Jv3 = sp.Matrix.hstack(pr3.diff(t1), pr3.diff(t2), 
    pr3.diff(t3))[0:3,:]    # Jv3: Velocity (Pr3)
# Get the velocity jacobian using (2nd for example)
# print(sp.latex(sp.simplify(Jv2).subs(sh_subs)))

Jw1 = sp.Matrix.hstack(pr1.diff(t1), pr1.diff(t2), 
    pr1.diff(t3))[3:6,:]    # Jw1: Angular Velocity (Pr1)
Jw2 = sp.Matrix.hstack(pr2.diff(t1), pr2.diff(t2), 
    pr2.diff(t3))[3:6,:]    # Jw2: Angular Velocity (Pr2)
Jw3 = sp.Matrix.hstack(pr3.diff(t1), pr3.diff(t2), 
    pr3.diff(t3))[3:6,:]    # Jw3: Angular Velocity (Pr3)
# Get the angular velocity jacobian using (2nd for example)
# print(sp.latex(sp.simplify(Jw2).subs(sh_subs)))

# %% Mass matrix
M = m1 * Jv1.T * Jv1 + Jw1.T * Icc1 * Jw1 + \
    m2 * Jv2.T * Jv2 + Jw2.T * Icc2 * Jw2 + \
    m3 * Jv3.T * Jv3 + Jw3.T * Icc3 * Jw3
# Get cell values using (2nd row, 3rd column example)
# print(sp.latex(sp.simplify(M[1,2]).subs(sh_subs)))
M = sp.simplify(M)

# %% Time dependent symbols
t = sp.Symbol("t")
q1 = sp.Function(r"q_1")(t)
q2 = sp.Function(r"q_2")(t)
q3 = sp.Function(r"q_3")(t)
q = sp.Matrix([[q1], [q2], [q3]])
q_dot = q.diff(t)
M_t = M.subs({t1:q1, t2:q2, t3:q3})
M_dot = M_t.diff(t) # Time derivative for mass matrix

# %% Coriolis and Centripetal Matrix
qdt_Mdiff = sp.Matrix.vstack(
    q_dot.T * M_t.diff(q1),
    q_dot.T * M_t.diff(q2), 
    q_dot.T * M_t.diff(q3))     # q.T * diff(M, q)
C_q_qdot = sp.simplify(M_dot - (1/2) * qdt_Mdiff)

# %% Gravity Vector
g = sp.symbols(r"g")
U = -(m1*g*pr1[1] + m2*g*pr2[1] + m3*g*pr3[1])
U_t = sp.simplify(U.subs({t1:q1, t2:q2, t3:q3}))    # Potential energy
U_tm = sp.Matrix([U_t]) # As a 1 element matrix
G = sp.Matrix.vstack(U_tm.diff(q1), U_tm.diff(q2), U_tm.diff(q3))

# %% Final torque equation
q_ddot = q_dot.diff(t)   # Second time derivative
tau = sp.simplify(M_t * q_ddot + C_q_qdot * q_dot + G)

# %% Shorthand subs
sh_subs = {
    q1.diff(t): sp.symbols(r"\dot{q}_1"),
    q2.diff(t): sp.symbols(r"\dot{q}_2"),
    q3.diff(t): sp.symbols(r"\dot{q}_3"),
    q1.diff(t, 2): sp.symbols(r"\ddot{q}_1"),
    q2.diff(t, 2): sp.symbols(r"\ddot{q}_2"),
    q3.diff(t, 2): sp.symbols(r"\ddot{q}_3"),
    q1: sp.symbols(r"q_1"),
    q2: sp.symbols(r"q_2"),
    q3: sp.symbols(r"q_3"),
    sp.sin(q1): sp.symbols(r"s_1"),
    sp.cos(q1): sp.symbols(r"c_1"),
    sp.sin(q2): sp.symbols(r"s_2"),
    sp.cos(q2): sp.symbols(r"c_2"),
    sp.sin(q3): sp.symbols(r"s_3"),
    sp.cos(q3): sp.symbols(r"c_3"),
    sp.sin(q1+q2): sp.symbols(r"s_{12}"),
    sp.cos(q1+q2): sp.symbols(r"c_{12}"),
    sp.sin(q2+q3): sp.symbols(r"s_{23}"),
    sp.cos(q2+q3): sp.symbols(r"c_{23}"),
    sp.sin(q1+q2+q3): sp.symbols(r"s_{123}"),
    sp.cos(q1+q2+q3): sp.symbols(r"c_{123}")
}
# Get Coriolis and Centripetal Matrix (2nd row, 3rd column example)
# print(sp.latex(sp.simplify(C_q_qdot[1,2].subs(sh_subs))))

# Get the Potential energy using
# print(sp.latex(U_t.subs(sh_subs)))

# Get the Gravity vector using (2nd element example)
# print(sp.latex(G[1].subs(sh_subs)))

# Get the torque / effort using (2nd element example)
# print(sp.latex(tau[1].subs(sh_subs)))

# %% Question 3.2: Check whether M is symmetric
if M_t.T - M_t == sp.Matrix(np.zeros((3,3))):
    print("The mass matrix is symmetric (M = M.T)")

# %% Question 3.3: Check if M_dot - 2 * C is skew symmetric
ssm = sp.simplify(M_dot - 2 * C_q_qdot)
if sp.simplify(ssm.T + ssm) == sp.Matrix(np.zeros((3,3))):
    print(f"M_dot - 2C is skew symmetric")
else:
    print(f"M_dot - 2C is not skew symmetric")
    sum_val = sp.simplify(ssm + ssm.T).subs(sh_subs)
    try:
        display(sum_val)
        # Get output using (2nd row, 3rd col as example)
        # print(sp.latex(ssm[1, 2].subs(sh_subs)))
    except:
        print(sum_val)

# %% C using cristoffel symbols
def cris_symb(i, j, k):
    def term(i, j, k):
        return M_t[i, j].diff(q[k])
    return term(i, j, k) + term(k, i, j) - term(k, j, i)

def c_ij(i, j):
    return (1/2) * ( \
            cris_symb(i, j, 0) * q_dot[0] + \
            cris_symb(i, j, 1) * q_dot[1] + \
            cris_symb(i, j, 2) * q_dot[2])


C_cris = sp.simplify(sp.Matrix([
    [c_ij(0, 0), c_ij(0, 1), c_ij(0, 2)],
    [c_ij(1, 0), c_ij(1, 1), c_ij(1, 2)],
    [c_ij(2, 0), c_ij(2, 1), c_ij(2, 2)],
]))
# Get output using (2nd row, 3rd col as example)
# print(sp.latex(C_cris[1, 2].subs(sh_subs)))
ssm_cris = sp.simplify(M_dot - 2 * C_cris)
if sp.simplify(ssm_cris.T + ssm_cris) == sp.Matrix(np.zeros((3,3))):
    print(f"M_dot - 2C_cris is skew symmetric")
else:
    print(f"M_dot - 2C_cris is not skew symmetric")
    display(sp.simplify(ssm_cris + ssm_cris.T).subs(sh_subs))
    # Get output using (2nd row, 3rd col as example)
    # print(sp.latex(ssm_cris[1, 2].subs(sh_subs)))


# %% Experimental section


# %%
tau_cris = sp.simplify(M_t * q_ddot + C_cris * q_dot + G)

# %%
