# %% Import everything
import numpy as np

# %% Function definitions
# Convert euler angles to rotation matrix
def euzyx_to_rotm(az, ay, ax):
    """
    Convert Euler ZYX angles to Rotation Matrix
    Parameters:
    - az: Along Z axis
    - ay: Along Y axis
    - ax: Along X axis
    Returns:
    - rot_m: A 3x3 rotation matrix
    """
    ca, sa = np.cos(az), np.sin(az) # Angle: Z
    cb, sb = np.cos(ay), np.sin(ay) # Angle: Y
    cg, sg = np.cos(ax), np.sin(ax) # Angle: X
    rot_m = np.array([
        [ca*cb, -sa*cg+sb*sg*ca, sa*sg+sb*ca*cg],
        [sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-sg*ca],
        [-sb, sg*cb, cb*cg]
    ], dtype=float)
    return rot_m

# %%
