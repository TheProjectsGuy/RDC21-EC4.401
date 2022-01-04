# %% Import everything
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patch
from fk_3r import jfk_min_3r, min_to_se2

# %% Parameters for robot
l1, l2, l3 = 4, 2, 1
axlim = [-8, 8]
jfk_min = lambda t1, t2, t3: jfk_min_3r(t1, t2, t3, l1, l2, l3)
fk_min = lambda t1, t2, t3: jfk_min(t1, t2, t3)[0]
t1, t2, t3 = map(float, np.deg2rad([0, 0, 0]))  # Joint angles (deg)

# %% Show in figure
ef, j3, j2 = jfk_min(t1, t2, t3)
# Workspace circle radius (outer and inner)
wo_cr = abs(fk_min(0, 0, 0)[0])
wi_cr = abs(fk_min(0, np.pi, 0)[0])
print(f"Work space: Outer: {wo_cr:.3f}, Inner: {wi_cr:.3f}")
# Dexterous workspace circle radius (outer and inner)
dwo_cr = abs(fk_min(0, 0, np.pi)[0])
dwi_cr = abs(fk_min(0, np.pi, np.pi)[0])
print(f"Dexterous space: Outer: {dwo_cr:.3f}, Inner: {dwi_cr:.3f}")
fig = plt.figure("3R FK", (8, 8))
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
# Workspaces
ax.add_patch(patch.Circle((0, 0), wo_cr, fill=False, ec='k',
    label="Workspace"))
ax.add_patch(patch.Circle((0, 0), wi_cr, fill=False, ec='k'))
ax.add_patch(patch.Circle((0, 0), dwo_cr, fill=False, ec='r',
    label="Dexterous WS"))
ax.add_patch(patch.Circle((0, 0), dwi_cr, fill=False, ec='r'))
ax.plot(0, 0, 'bo', fillstyle='none')
ax.plot([0, j2[0]], [0, j2[1]], 'k')
ax.plot(j2[0], j2[1], 'co', fillstyle='none')
ax.plot([j2[0], j3[0]], [j2[1], j3[1]], 'k')
ax.plot(j3[0], j3[1], 'mo', fillstyle='none')
ax.plot([j3[0], ef[0]], [j3[1], ef[1]], 'k')
ax.plot(ef[0], ef[1], 'go')
ax.set_xlim(axlim)
ax.set_ylim(axlim)
ax.legend()
# fig.savefig("ex2-2-dws-inner.png", dpi=300)
plt.show()

# %%
