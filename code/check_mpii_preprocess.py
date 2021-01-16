import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, img_as_ubyte, img_as_float32
from os.path import join
import numpy as np

index_to_check = 16100
image_dir = "D://Brown//Senior//CSCI_1470//FINAL//MPII"
img = mpimg.imread(image_dir + '/cropped_mpii/' + str(index_to_check) + '.png')

f = open(image_dir + '/joints.txt', "r")
contents = f.readlines()
num_imgs = int(len(contents)/14)
annots = np.zeros((num_imgs, 14, 3))

for i in range(num_imgs):
    for j in range(14):
        string = contents[i * 14 + j]
        split_str = np.array(string.split())
        split_str = split_str.astype(np.float)
        annots[i][j] = split_str
imgplot = plt.imshow(img)
annots_indexed = annots[index_to_check]
x1 = annots_indexed[:,0]
y1 = annots_indexed[:,1]
plt.scatter(x1, y1)
plt.show()