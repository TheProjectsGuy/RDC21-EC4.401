# %% Import everything
from matplotlib import pyplot as plt
from matplotlib import patches as patch
from matplotlib import widgets as wd
import numpy as np
# Kinematics
from ik_3r import ik_3r
from fk_3r import jfk_min_3r

# %% Variables
l1, l2, l3 = map(float, [2, 1, 0.5])    # Link lengths
# Line properties
ln_start = [1.1, 2]     # Starting point (x, y)
ln_end = [2, -1]        # Ending point (x, y)
ln_th = np.deg2rad(40)  # Angle to maintain throughout
num_ts = 50             # Number of timesteps
# Minimal representation poses
jfk_min = lambda t1, t2, t3: jfk_min_3r(t1, t2, t3, l1, l2, l3)
fk_min = lambda t1, t2, t3: jfk_min(t1, t2, t3)[0]
# Inverse Kinematics
ik_min = lambda x, y, th: ik_3r(x, y, th, l1, l2, l3)
# Axis limit
lims = [-(l1+l2+l3)*1.1, (l1+l2+l3)*1.1]
# Interpolate
t = np.linspace(0, 1, num_ts)
ln_t = np.vstack((
    np.array(ln_start)[0] * (1-t) + np.array(ln_end)[0] * t,
    np.array(ln_start)[1] * (1-t) + np.array(ln_end)[1] * t))

# %% Show in figure
fig = plt.figure("3R IK", (8, 8))
plt.subplots_adjust(bottom=0.2)
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
# Limits
ax.grid()
ax.set_xlim(lims)
ax.set_ylim(lims)
# Reachable and dexterous workspace
wo_cr = abs(fk_min(0, 0, 0)[0])
wi_cr = abs(fk_min(0, np.pi, 0)[0])
dwo_cr = abs(fk_min(0, 0, np.pi)[0])
dwi_cr = abs(fk_min(0, np.pi, np.pi)[0])
ax.add_patch(patch.Circle((0, 0), wo_cr, fill=False, ec='k'))
ax.add_patch(patch.Circle((0, 0), wi_cr, fill=False, ec='k'))
ax.add_patch(patch.Circle((0, 0), dwo_cr, fill=False, ec='r'))
ax.add_patch(patch.Circle((0, 0), dwi_cr, fill=False, ec='r'))
# Plot line
ax.plot(ln_start[0], ln_start[1], 'yx',
    ln_end[0], ln_end[1], 'yx', ms=10)
ax.plot(ln_t[0], ln_t[1], 'y--')
# Inverse kinematics
x, y, th = ln_start[0], ln_start[1], ln_th
a1, a2, a3 = ik_min(x, y, th)
ef_p, j3_p, j2_p = jfk_min(a1, a2, a3)
pj1, pj2, pj3 = ax.plot(    # Joints
    [0], [0], 'bo',
    [j2_p[0]], [j2_p[1]], 'co',
    [j3_p[0]], [j3_p[1]], 'mo', fillstyle='none'
)
pef, = ax.plot([ef_p[0]], [ef_p[1]], 'go') # End effector
bl1, bl2, bl3 = ax.plot(    # Body links
    [0, j2_p[0]], [0, j2_p[1]], 'k-',
    [j2_p[0], j3_p[0]], [j2_p[1], j3_p[1]], 'k-',
    [j3_p[0], ef_p[0]], [j3_p[1], ef_p[1]], 'k-'
)

# %% Graphics object
ts = [1, len(t)]
axcolor = 'lightgoldenrodyellow'
axj1 = plt.axes([0.19, 0.1, 0.65, 0.03], fc=axcolor)
sts = wd.Slider(axj1, "T", ts[0], ts[1], valinit=0, valstep=1)
# Update function for graphics handle
def on_slider_update(tsn_f):
    tsn = int(tsn_f) - 1
    # Target point
    x, y, th = ln_t[0][tsn], ln_t[1][tsn], ln_th
    a1, a2, a3 = ik_min(x, y, th)   # Angles
    ef_p, j3_p, j2_p = jfk_min(a1, a2, a3)  # FK
    # Update points
    pj2.set_xdata([j2_p[0]])
    pj2.set_ydata([j2_p[1]])
    pj3.set_xdata([j3_p[0]])
    pj3.set_ydata([j3_p[1]])
    pef.set_xdata([ef_p[0]])
    pef.set_ydata([ef_p[1]])
    # Body links
    bl1.set_xdata([0, j2_p[0]])
    bl1.set_ydata([0, j2_p[1]])
    bl2.set_xdata([j2_p[0], j3_p[0]])
    bl2.set_ydata([j2_p[1], j3_p[1]])
    bl3.set_xdata([j3_p[0], ef_p[0]])
    bl3.set_ydata([j3_p[1], ef_p[1]])
    # Update render
    fig.canvas.draw_idle()
# Set update function
sts.on_changed(on_slider_update)

# %% Main plot
plt.show()

# %%
