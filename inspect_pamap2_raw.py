from common.data import Opportunity, OpportunityOptions, OpportunityView, Pamap2, Pamap2Options, Pamap2View
import matplotlib.pyplot as plt
import matplotlib as mpl

validation_data = Pamap2(
  root='/home/jonas/Stuff/Datasets/PAMAP2_Dataset',
  window=300,
  stride=300,
  opts=[Pamap2Options.ALL_SUBJECTS, Pamap2Options.OPTIONAL1]
  )

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

view1 = Pamap2View(['imu_ankle'])
view2 = Pamap2View(['imu_hand'])
segment1 = view1(segments1[12])
segment2 = view1(segments2[12])
segment3 = view1(segments3[12])
segment4 = view2(segments3[12])

fig_a, ax_a = plt.subplots(3, 1)
for a in ax_a:
  a.set_ylabel("$\Delta V$")
  a.set_xlabel("$T$")
  a.set_yticks([])
  a.set_xticks([])

ax_a[0].plot(segment1[:,1])
ax_a[1].plot(segment2[:,1])
ax_a[2].plot(segment3[:,1])
fig_a.tight_layout()


fig_b, ax_b = plt.subplots(9,1)
colors = mpl.colormaps['Accent']
for i, a in enumerate(ax_b):
  a.set_yticks([])
  a.set_xticks([])
  a.plot(segment3[:,1+i], color=colors(i/9))
fig_b.tight_layout()

fig_c, ax_c = plt.subplots(9,1)
for i, a in enumerate(ax_c):
  a.set_yticks([])
  a.set_xticks([])
  a.plot(segment4[:,1+i], color=colors(i/9))
fig_c.tight_layout()

plt.show(block=True)
