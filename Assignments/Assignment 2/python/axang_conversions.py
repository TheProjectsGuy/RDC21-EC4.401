# %% Import everything
import numpy as np

# %% Function definitions

# Rotation matrix to Axis angle
def rotm_to_axang(rot_m):
    """
    Convert rotation matrix to axis-angle representation
    Parameters:
    - rot_m: np.ndarray     shape: (3, 3)
        Rotation Matrix
    Returns:
    - ax: np.ndarray    shape: (3,1)
        The axis of rotation
    - ang: float
        The angle in radians
    """
    # Parse elements
    r11 = rot_m[0][0]
    r12 = rot_m[0][1]
    r13 = rot_m[0][2]
    r21 = rot_m[1][0]
    r22 = rot_m[1][1]
    r23 = rot_m[1][2]
    r31 = rot_m[2][0]
    r32 = rot_m[2][1]
    r33 = rot_m[2][2]
    # Angle
    ang_rad = np.arccos((r11+r22+r33-1)/2)
    # Axis
    ax = (1/(2*np.sin(ang_rad))) * np.array([
        [r32-r23],
        [r13-r31],
        [r21-r12]
    ])
    return ax.flatten(), ang_rad

# Axis-angle to rotation matrix
def axang_to_rotm(ax, ang):
    """
    Convert axis-angle representation to rotation matrix
    Parameters:
    - ax: np.ndarray    shape: (3,)
        Axis of rotation (should be normalized beforehand)
    - ang: float
        The angle of rotation (in radians) using right-hand thumb rule
    Returns:
    - rot_m: np.ndarray     shape: (3, 3)
        The resultant rotation matrix
    """
    nx, ny, nz = map(float, ax)
    st, ct = np.sin(ang), np.cos(ang)
    vt = 1-ct  # For brevity
    rot_m = np.array([
        [1+vt*(nx**2-1), -nz*st+vt*nx*ny, ny*st+vt*nx*nz],
        [nz*st+vt*ny*nx, 1+vt*(ny**2-1), -nx*st+vt*ny*nz],
        [-ny*st+vt*nz*nx, nx*st+vt*nz*ny, 1+vt*(nz**2-1)]
    ])
    return rot_m

# %% Main test
if __name__ == "__main__":
    ax, ang = np.array([1, 0, 0]), np.deg2rad(45)
    ax = ax/np.linalg.norm(ax)
    rot_m = axang_to_rotm(ax, ang)
    ax2, ang2 = rotm_to_axang(rot_m)
    if np.allclose(ax, ax2) and np.allclose(ang, ang2):
        print("Functions seem to map correctly")

# %%
