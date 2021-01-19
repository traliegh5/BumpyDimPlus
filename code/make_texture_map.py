
## Load .pkl 
working_dir = "D:\Brown\Senior\CSCI_1470\FINAL\smpl_UV"
filename_bary = working_dir + "\\uv_bary"
uv = load_obj(filename_bary)

#### Load image
image_dir = "D://Brown//Senior//CSCI_1470//FINAL//MPII//cropped_mpii"
image = mpimg.imread(image_dir + '//05681.png')

## Make start model
star = STAR(gender='neutral')
trans = tf.constant(np.random.rand(1,3), dtype=tf.float32)
pose = tf.constant(np.zeros((1,72)),dtype=tf.float32)
betas = tf.constant(np.zeros((1,10)),dtype=tf.float32)
m = star(pose,betas,trans)

bary_map = tf.convert_to_tensor(np.array(uv), dtype=tf.float32)
out_img = map(m[0], tf.convert_to_tensor(star.f, dtype=tf.int32), bary_map, image, trans)

tex_map = np.zeros((1024,1024,3))