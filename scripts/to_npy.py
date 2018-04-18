from PIL import Image
import numpy as np
import os


IN_DIR = 'resized_maps'
OUT_DIR = 'resized_maps_npy'

img_names = os.listdir(IN_DIR)
for name in img_names:
    img = Image.open(os.path.join(IN_DIR, name))
    arr = np.array(img)
    np.save('{}/{}.npy'.format(OUT_DIR, name), arr)

