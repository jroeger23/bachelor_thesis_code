from common.data import Opportunity, OpportunityOptions, OpportunityView, Pamap2, Pamap2Options, Pamap2View
import matplotlib.pyplot as plt

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

view = Pamap2View(['imu_ankle'])
segment1 = view(segments1[12])
segment2 = view(segments2[12])
segment3 = view(segments3[12])

fig, ax = plt.subplots(3, 1)
for a in ax:
  a.set_ylabel("$\Delta V$")
  a.set_xlabel("$T$")
  a.set_yticks([])
  a.set_xticks([])

ax[0].set_title(Pamap2View.describeLabels(label1))
ax[0].plot(segment1[:,1])
ax[1].set_title(Pamap2View.describeLabels(label2))
ax[1].plot(segment2[:,1])
ax[2].set_title(Pamap2View.describeLabels(label3))
ax[2].plot(segment3[:,1])
fig.tight_layout()
plt.show(block=True)
