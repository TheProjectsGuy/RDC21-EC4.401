# %% Import everything
import numpy as np

# %% Functions
# IK of 3R
def ik_3r(x, y, al, l1=2, l2=1, l3=0.5):
    """
    The inverse kinematics of a 3R manipulator

    Parameters:
    - x, y, al: The point (x, y) and pose (al) in the plane
    - l1, l2, l3    default: (2, 1, 0.5) respectively
        The link lengths
    
    Returns:

    """
    xj = x - l3*np.cos(al)
    yj = y - l3*np.sin(al)
    # Angles
    th2 = np.arccos((xj**2+yj**2-l1**2-l2**2)/(2*l1*l2))
    th1 = np.arctan2(yj, xj) - \
        np.arctan2(l2*np.sin(th2), l1+l2*np.cos(th2))
    th3 = al - th1 - th2
    # Return angles
    return th1, th2, th3

# %%
