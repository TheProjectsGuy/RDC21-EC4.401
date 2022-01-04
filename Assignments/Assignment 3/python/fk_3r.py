# %% Import everything
import numpy as np

# %% Function definitions
# Forward Kinematics of EF and Joints of 3R manipulator
def jfk_min_3r(t1, t2, t3, l1, l2, l3):
    """
    Return the Forward Kinematics, with joint positions as well. This
    is helpful when plotting.

    Parameters:
    - t1, t2, t3: float(s)
        The joint angles (in radians)
    - l1, l2, l3: float(s)
        The link lengths
    Returns:
    - ef_min: np.ndarray     shape: (3,)
        The (x, y, theta) pose of the end effector
    - j3_min: np.ndarray     shape: (3,)
        The (x, y, theta) pose of the 3rd joint (link 2 to 3)
    - j2_min: np.ndarray     shape: (3,)
        The (x, y, theta) pose of the 3rd joint (link 1 to 2)
    """
    # Joint 2
    j2_min = np.array([l1*np.cos(t1), l1*np.sin(t1), t1+t2])
    # Joint 3
    j3_min = np.array([
        l1*np.cos(t1) + l2*np.cos(t1+t2),
        l1*np.sin(t1) + l2*np.sin(t1+t2),
        t1+t2+t3
    ])
    # End effector
    ef_min = np.array([
        l1*np.cos(t1) + l2*np.cos(t1+t2) + l3*np.cos(t1+t2+t3),
        l1*np.sin(t1) + l2*np.sin(t1+t2) + l3*np.sin(t1+t2+t3),
        t1+t2+t3
    ])
    return ef_min, j3_min, j2_min

# %%
