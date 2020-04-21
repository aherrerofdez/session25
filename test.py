from matplotlib import pylab as plt

series1 = []
series2 = []
series3 = []

for i in range(0, 30):
    series1.append(i)
    series2 += [i*i]
    series3 += [i**3]

plt.figure("first")
plt.plot(list(range(0, 30)), series1, "r+", label="Linear", linewidth=2, ms=10)
# plt.figure("second")
plt.plot(list(range(0, 30)), series2, "k:^", label="Quadric", linewidth=0.5)
plt.legend(loc="upper left")
plt.figure("third")
plt.plot(list(range(0, 30)), series3)

plt.figure("first")
plt.title("Linear")
plt.ylim(0, 1000)
plt.xlabel("Series")
plt.ylabel("Linear Progression")
# plt.figure("second")
plt.title("Quadric")

plt.figure("third")
plt.title("Cubic")
plt.ylabel("Cubic Progression")

plt.show()
