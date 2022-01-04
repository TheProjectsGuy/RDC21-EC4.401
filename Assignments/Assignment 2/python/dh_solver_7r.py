# %% Import everything
import numpy as np
import sympy as sp

# %% Functions
# Return DH Parameters
def DH_tf(a, al, d, th):
    """
    Returns the 4x4 homogeneous transformation matrix, given the four
    modified DH parameters.
    Parameters:
    - a: float      Link length (i-1)
    - al: float     Link twist (i-1)
    - d: float      Joint offset (i)
    - th: float     Joint angle / twist (i)
    Returns
    - htf: sp.Matrix    shape: (4, 4)
        A homogeneous transformation matrix
    """
    c = sp.cos
    s = sp.sin
    return sp.Matrix([
        [c(th), -s(th), 0, a],
        [c(al)*s(th), c(al)*c(th), -s(al), -d*s(al)],
        [s(al)*s(th), s(al)*c(th), c(al), d*c(al)],
        [0, 0, 0, 1]
    ])

# %%
# Joint variables
t1, t2, t3 = sp.symbols(r'\theta_1, \theta_2, \theta_3')
t4, t5, t6 = sp.symbols(r'\theta_4, \theta_5, \theta_6')
t7 = sp.symbols(r"\theta_7")
# Link offsets (joint offsets in DH)
l1, l2, l3, l4 = sp.symbols(r"L_1, L_2, L_3, L_4")
# Some symbols
pi = sp.pi
pi_2 = pi/2

# %% DH Parameters
dh_params = [
    [0, 0, l1, t1],
    [0, pi_2, 0, pi+t2],
    [0, pi_2, l2, pi+t3],
    [0, pi_2, 0, pi+t4],
    [0, pi_2, l3, pi+t5],
    [0, pi_2, 0, pi+t6],
    [0, pi_2, 0, t7],
    [0, 0, l4, 0]
]

# %% Transforms
tf_0_1 = DH_tf(*dh_params[0])
tf_1_2 = DH_tf(*dh_params[1])
tf_2_3 = DH_tf(*dh_params[2])
tf_3_4 = DH_tf(*dh_params[3])
tf_4_5 = DH_tf(*dh_params[4])
tf_5_6 = DH_tf(*dh_params[5])
tf_6_7 = DH_tf(*dh_params[6])
tf_7_8 = DH_tf(*dh_params[7])

# %% Transformations in home reference frame
tf_0_2 = sp.simplify(tf_0_1 * tf_1_2)
tf_0_3 = sp.simplify(tf_0_2 * tf_2_3)
tf_0_4 = sp.simplify(tf_0_3 * tf_3_4)
tf_0_5 = sp.simplify(tf_0_4 * tf_4_5)
tf_0_6 = sp.simplify(tf_0_5 * tf_5_6)
tf_0_7 = sp.simplify(tf_0_6 * tf_6_7)
tf_0_8 = sp.simplify(tf_0_7 * tf_7_8)

# %% Home position verification
subs = {
    t1:0, t2: 0, t3: 0, t4: 0, t5: 0, t6: 0, t7: 0
}
home_pose_7 = sp.simplify(tf_0_7.subs(subs))
home_pose_8 = sp.simplify(tf_0_8.subs(subs))

# %% Short hand substitutions
sh_subs = {
    sp.sin(t1): sp.symbols("s_1"),
    sp.cos(t1): sp.symbols("c_1"),
    sp.sin(t2): sp.symbols("s_2"),
    sp.cos(t2): sp.symbols("c_2"),
    sp.sin(t3): sp.symbols("s_3"),
    sp.cos(t3): sp.symbols("c_3"),
    sp.sin(t4): sp.symbols("s_4"),
    sp.cos(t4): sp.symbols("c_4"),
    sp.sin(t5): sp.symbols("s_5"),
    sp.cos(t5): sp.symbols("c_5"),
    sp.sin(t6): sp.symbols("s_6"),
    sp.cos(t6): sp.symbols("c_6"),
    sp.sin(t7): sp.symbols("s_7"),
    sp.cos(t7): sp.symbols("c_7"),
}
tf_0_7_sh = sp.simplify(tf_0_7.subs(sh_subs))
tf_0_8_sh = sp.simplify(tf_0_8.subs(sh_subs))

# %%
