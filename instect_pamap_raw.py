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

fig, ax = plt.subplots(3, 3)
for row in ax:
  for a in row:
    a.set_ylabel("$[\\frac{m}{s^2}]$")
    a.set_xlabel("$[s]$")
    a.set_yticks([])
    a.set_xticks([])
ax[0][0].set_title(Pamap2View.describeLabels(label1))
ax[0][0].plot(segment1[:,1])
ax[1][0].plot(segment1[:,2])
ax[2][0].plot(segment1[:,3])
ax[0][1].set_title(Pamap2View.describeLabels(label2))
ax[0][1].plot(segment2[:,1])
ax[1][1].plot(segment2[:,2])
ax[2][1].plot(segment2[:,3])
ax[0][2].set_title(Pamap2View.describeLabels(label3))
ax[0][2].plot(segment3[:,1])
ax[1][2].plot(segment3[:,2])
ax[2][2].plot(segment3[:,3])
plt.show(block=True)
