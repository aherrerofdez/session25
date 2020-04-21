from matplotlib import pylab as plt

xaxis = plt.arange(0, plt.pi*4, 0.1)

plt.figure("sin() and cos() animation")

for i in xaxis:
    series1 = plt.sin(xaxis - i)
    series2 = plt.cos(xaxis - i)
    # plt.subplot(211)
    plt.plot(xaxis, series1, "--", label="sin()")
    # plt.subplot(212)
    plt.plot(xaxis, series2, ":", label="cos()")
    plt.legend()
    plt.draw()  #instead of plt.show()
    plt.pause(0.025)
    plt.clf()