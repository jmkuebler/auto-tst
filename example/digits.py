from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

import autotst
from autotst.model import AutoGluonTabularPredictor, AutoGluonImagePredictor

import os
from PIL import Image

digits = load_digits(n_class=2)  # contains 8x8 pixel images of digits
digits_images = digits.images
zeros = digits_images[digits.target == 0]
ones = digits_images[digits.target == 1]
images_sorted = np.concatenate((zeros, ones))

# # Use something like this to show images.
# plt.gray()
# plt.matshow(ones[0])
# plt.show()

# generate two different distributions.
data_P = np.array(images_sorted[np.random.randint(0, len(zeros), size=20)])
data_Q = np.array(images_sorted[np.random.randint(len(zeros), len(zeros)+len(ones), size=20)])


# ======= Run default ========
# need to flatten images for tabular predictor
tst_new = autotst.AutoTST(data_P.reshape((len(data_P), -1)), data_Q.reshape((len(data_Q), -1)))
print("P-value with default settings and tabular predictor (time limit of 1 minutes): ", tst_new.p_value())

# ===== run test with AutoGluonImagePredictor ========

# AutoGluonImagePredictor requires the images to be stored in files
if not os.path.exists('temp_images'):
    os.mkdir('temp_images')
paths_P = []
for i in range(len(data_P)):
    im = Image.fromarray((data_P[i]*255).astype(np.uint8))
    im = im.convert("L")
    path = f'temp_images/im_P_{i}.png'
    im.save(path)
    paths_P.append(path)
paths_Q = []
for i in range(len(data_Q)):
    im = Image.fromarray((data_Q[i]*255).astype(np.uint8))
    im = im.convert("L")
    path = f'temp_images/im_Q_{i}.png'
    im.save(path)
    paths_Q.append(path)

# We pass a list of image paths of both samples
tst_img = autotst.AutoTST(paths_P, paths_Q, model=AutoGluonImagePredictor)
tst_img.split_data()
tst_img.fit_witness(time_limit=60)
print("P-value with Image predictor  ", tst_img.p_value_evaluate())
print(tst_img.interpret(1))
