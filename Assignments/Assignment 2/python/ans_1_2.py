# %% Import everything
import numpy as np

# %% Define Functions
# Convert rotation matrix to ruler angles
def rotm_to_euzyx(rot_m):
    """
    Convert Rotation Matrix to Euler ZYX angles. In case of
    singularity, the rotation about Z is assumed to be 0.
    Paraemters:
    - rot_m: A 3x3 SO(3) Rotation Matrix
    Returns:
    - az: Angle of rotation about Z axis (radians)
    - ay: Angle of rotation about Y axis (radians)
    - ax: Angle of rotation about X axis (radians)
    """
    # Angles
    az, ay, ax = None, None, None
    # Procedure
    if (np.isclose(rot_m[2][0], -1)):   # Beta = 90 deg
        az = 0  # Alpha = 0
        ay = np.pi/2
        ax = -np.arctan2(rot_m[1][2], rot_m[0][2])
    elif (np.isclose(rot_m[2][0], 1)):  # Beta = -90 deg
        az = 0  # Alpha = 0
        ay = -np.pi/2
        ax = np.arctan2(-rot_m[1][2], -rot_m[0][2])
    else:   # General case: Not singularity
        az = np.arctan2(rot_m[1][0], rot_m[0][0])
        ay = np.arctan2(-rot_m[2][0], 
            np.sqrt(rot_m[0][0]**2+rot_m[1][0]**2))
        ax = np.arctan2(rot_m[2][1], rot_m[2][2])
    return az, ay, ax

# %%
