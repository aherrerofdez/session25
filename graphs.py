from matplotlib import pylab as plt

# xseries = list(range(0, 30)) -> replace for better plot of sin() function

xseries = plt.arange(0, 30, 0.1)

# print(xseries)

series1 = []
series2 = []
series3 = plt.sin(xseries)

maxvalues_xaxis = plt.arange(plt.pi/2, 30, plt.pi)
maxvalues_yaxis = plt.sin(maxvalues_xaxis)

for i in xseries:
    series1.append(i)
    series2.append(i/2)

# Create individual figures using subplot()
plt.subplot(211)  # xyz -> row, column, position
plt.plot(xseries, series1, label="Faster Linear Progression")
plt.plot(xseries, series2, "y:", label="Slower Linear Progression")
plt.legend()

plt.subplot(212)  # xyz -> row, column, position
# style(color(r, g, b, c, m, y, k), line(:, --), marker(^))
plt.plot(xseries, series3, "k--", label="sin() function")
plt.plot(maxvalues_xaxis, maxvalues_yaxis, "r^", label="Extreme Values of sin()")
# plt.text() is used to put text inside the graph
for i in range(len(maxvalues_xaxis)):
    plt.text(maxvalues_xaxis[i], maxvalues_yaxis[i], str(maxvalues_yaxis[i]))
plt.ylim(-2, 4)
plt.legend(loc="upper left")

plt.show()
