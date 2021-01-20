import matplotlib.pyplot as plt
import numpy as np

f = open("gen_loss.txt", "r")
contents_gen = f.readlines()
f.close()

f = open("dis_loss.txt", "r")
contents_dis = f.readlines()
f.close()

x_array = np.arange(len(contents_dis))

contents_gen = np.array(contents_gen, dtype=np.float32)
plt.plot(x_array, contents_gen, color='blue')

contents_dis = np.array(contents_dis, dtype=np.float32)
plt.plot(x_array, contents_dis, color='green')
plt.savefig('loss_plot.png')