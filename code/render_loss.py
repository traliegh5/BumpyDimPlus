import matplotlib.pyplot as plt

f = open(image_dir + "gen_loss.txt", "r")
contents_gen = f.readlines()
f.close()

f = open(image_dir + "dis_loss.txt", "r")
contents_dis = f.readlines()
f.close()

x_array = np.arange(len(contents_dis))

contents_gen = np.array(contents_gen)
plt.plot(x_array, contents_gen, color='blue')

contents_dis = np.array(contents_dis)
plt.plot(x_array, contents_dis, color='green')
plt.savefig('loss_plot.png')