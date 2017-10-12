import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

out = np.zeros([1084, 2300, 3], dtype = int)
out[0:1084, 0:2300, 0:3] = 255

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0008773_v6/tmp.jpg'
org_img = misc.imread(img_path)
[h, w, c] = org_img.shape
out[0:h, 0:w, 0:3] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0008773_v6/r_base.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[0:h, 460:(460+w), 0] = org_img
out[0:h, 460:(460+w), 1] = org_img
out[0:h, 460:(460+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0008773_v6/r_class.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[0:h, 920:(920+w), 0] = org_img
out[0:h, 920:(920+w), 1] = org_img
out[0:h, 920:(920+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0008773_v6/r_color.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[0:h, 1380:(1380+w), 0] = org_img
out[0:h, 1380:(1380+w), 1] = org_img
out[0:h, 1380:(1380+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0008773_v6/r_triplet.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[0:h, 1840:(1840+w), 0] = org_img
out[0:h, 1840:(1840+w), 1] = org_img
out[0:h, 1840:(1840+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0026247_v6/tmp.jpg'
org_img = misc.imread(img_path)
[h, w, c] = org_img.shape
out[540:540+h, 0:w, 0:3] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0026247_v6/r_base.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[540:540+h, 460:(460+w), 0] = org_img
out[540:540+h, 460:(460+w), 1] = org_img
out[540:540+h, 460:(460+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0026247_v6/r_class.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[540:540+h, 920:(920+w), 0] = org_img
out[540:540+h, 920:(920+w), 1] = org_img
out[540:540+h, 920:(920+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0026247_v6/r_color.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[540:540+h, 1380:(1380+w), 0] = org_img
out[540:540+h, 1380:(1380+w), 1] = org_img
out[540:540+h, 1380:(1380+w), 2] = org_img

img_path = '/Users/Winslow/Desktop/exp_res/RF-visualize/0026247_v6/r_triplet.jpg'
org_img = misc.imread(img_path)
[h, w] = org_img.shape
out[540:540+h, 1840:(1840+w), 0] = org_img
out[540:540+h, 1840:(1840+w), 1] = org_img
out[540:540+h, 1840:(1840+w), 2] = org_img

misc.imsave('/Users/Winslow/Desktop/tmp.jpg', out)
#print org_img
print h, w, c