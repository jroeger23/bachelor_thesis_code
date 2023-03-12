import matplotlib as mpl
import matplotlib.pyplot as plt

from common.data import Pamap2, Pamap2IMUView, Pamap2Options

validation_data = Pamap2(window=300,
                         stride=300,
                         opts=[Pamap2Options.ALL_SUBJECTS, Pamap2Options.OPTIONAL1])

segments1 = []
segments2 = []
segments3 = []
label1 = 4
label2 = 5
label3 = 6
for s, l in reversed(validation_data):
  if l == label1:
    segments1.append(s)
  if l == label2:
    segments2.append(s)
  if l == label3:
    segments3.append(s)

view1 = Pamap2IMUView(['imu_a'], with_heart_rate=False)
view2 = Pamap2IMUView(['imu_h'], with_heart_rate=False)
segment1, _ = view1(segments1[12], l)
segment2, _ = view1(segments2[12], l)
segment3, _ = view1(segments3[12], l)
segment4, _ = view2(segments3[12], l)

fig_a, ax_a = plt.subplots(3, 1)
for a in ax_a:
  a.set_ylabel("$\Delta V$")
  a.set_xlabel("$T$")
  a.set_yticks([])
  a.set_xticks([])

ax_a[0].plot(segment1[:, 3])
ax_a[1].plot(segment2[:, 3])
ax_a[2].plot(segment3[:, 3])
ax_a[0].set_title('Laufen')
ax_a[1].set_title('Rennen')
ax_a[2].set_title('Fahrrad fahren')
fig_a.tight_layout()

fig_b, ax_b = plt.subplots(9, 1)
colors = mpl.colormaps['Accent']
for i, a in enumerate(ax_b):
  a.set_yticks([])
  a.set_xticks([])
  a.plot(segment3[:, 1 + i], color=colors(i / 9))
fig_b.tight_layout()

fig_c, ax_c = plt.subplots(9, 1)
for i, a in enumerate(ax_c):
  a.set_yticks([])
  a.set_xticks([])
  a.plot(segment4[:, 1 + i], color=colors(i / 9))
fig_c.tight_layout()

plt.show(block=True)
