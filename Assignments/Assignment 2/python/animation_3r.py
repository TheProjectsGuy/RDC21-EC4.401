# %% Import everything
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patch
from matplotlib import widgets as wd
from fk_3r import jfk_min_3r, min_to_se2

# %% Robot variables
l1, l2, l3 = map(float, [4, 2, 1])
axlims = [-8, 8]
t1, t2, t3 = map(float, [0, 0, 0])
jfk_min = lambda t1, t2, t3: jfk_min_3r(t1, t2, t3, l1, l2, l3)
fk_min = lambda t1, t2, t3: jfk_min(t1, t2, t3)[0]

# %% Variables
fig = plt.figure("3R Manipulator", (8, 8))
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(axlims)
ax.set_ylim(axlims)
plt.subplots_adjust(bottom=0.33)
# Reachable and dexterous workspace
wo_cr = abs(fk_min(0, 0, 0)[0])
wi_cr = abs(fk_min(0, np.pi, 0)[0])
dwo_cr = abs(fk_min(0, 0, np.pi)[0])
dwi_cr = abs(fk_min(0, np.pi, np.pi)[0])
ax.add_patch(patch.Circle((0, 0), wo_cr, fill=False, ec='k'))
ax.add_patch(patch.Circle((0, 0), wi_cr, fill=False, ec='k'))
ax.add_patch(patch.Circle((0, 0), dwo_cr, fill=False, ec='r'))
ax.add_patch(patch.Circle((0, 0), dwi_cr, fill=False, ec='r'))
# Starting pose
ef_pose, j3_pose, j2_pose = jfk_min(t1, t2, t3)
p1, p2, p3 = ax.plot(  # Point to joints and end effector
    [0], [0], 'bo',
    [j2_pose[0]], [j2_pose[1]], 'co',
    [j3_pose[0]], [j3_pose[1]], 'mo', fillstyle='none'
)
pef, = ax.plot([ef_pose[0]], [ef_pose[1]], 'go')
bl1, bl2, bl3 = ax.plot(    # Body links
    [0, j2_pose[0]], [0, j2_pose[1]], 'k-',
    [j2_pose[0], j3_pose[0]], [j2_pose[1], j3_pose[1]], 'k-',
    [j3_pose[0], ef_pose[0]], [j3_pose[1], ef_pose[1]], 'k-'
)

# %% Graphics objects
jlim = [0, 2*np.pi]
axcolor = 'lightgoldenrodyellow'
axj1 = plt.axes([0.19, 0.2, 0.65, 0.03], fc=axcolor)
axj2 = plt.axes([0.19, 0.15, 0.65, 0.03], fc=axcolor)
axj3 = plt.axes([0.19, 0.1, 0.65, 0.03], fc=axcolor)
sj1 = wd.Slider(axj1, "J1", jlim[0], jlim[1], valinit=t1)
sj2 = wd.Slider(axj2, "J2", jlim[0], jlim[1], valinit=t2)
sj3 = wd.Slider(axj3, "J3", jlim[0], jlim[1], valinit=t3)
# Update function for graphics handle
def on_slider_change(val):
    # Calculate FK (from slider values)
    ef_pose, j3_pose, j2_pose = jfk_min(sj1.val, sj2.val, sj3.val)
    # Points
    p1.set_xdata([0])
    p1.set_ydata([0])
    p2.set_xdata([j2_pose[0]])
    p2.set_ydata([j2_pose[1]])
    p3.set_xdata([j3_pose[0]])
    p3.set_ydata([j3_pose[1]])
    pef.set_xdata([ef_pose[0]])
    pef.set_ydata([ef_pose[1]])
    # Body links
    bl1.set_xdata([0, j2_pose[0]])
    bl1.set_ydata([0, j2_pose[1]])
    bl2.set_xdata([j2_pose[0], j3_pose[0]])
    bl2.set_ydata([j2_pose[1], j3_pose[1]])
    bl3.set_xdata([j3_pose[0], ef_pose[0]])
    bl3.set_ydata([j3_pose[1], ef_pose[1]])
    # Update render
    fig.canvas.draw_idle()
# Set the update function
sj1.on_changed(on_slider_change)
sj2.on_changed(on_slider_change)
sj3.on_changed(on_slider_change)

# %% Main plot
plt.show()
