# %% Import everything
import numpy as np

# %% Function definitions
# Convert minimal representation to SE2
def min_to_se2(min_repr):
    """
    Converts a minimal representation (x, y, theta) to a 3x3
    homogeneous transformation matrix SE(2)
    Parameters:
    - min_repr: np.ndarray or list  shape: (3,)
        The values of (x, y, theta)
    Returns:
    - se2_repr: np.ndarray      shape: (3, 3)
        The SE(2) representation
    """
    x, y, theta = list(map(float, min_repr))
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])


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
