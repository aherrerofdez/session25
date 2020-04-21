from matplotlib import pylab as plt

series1 = []
series2 = []
series3 = []

for i in range(0, 30):
    series1.append(i)
    series2 += [i*i]
    series3 += [i**3]

plt.plot(list(range(0, 30)), series1)
plt.plot(list(range(0, 30)), series2)
plt.plot(list(range(0, 30)), series3)

plt.show()
