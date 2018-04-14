import numpy as np
import os
from skimage import io
import json

if __name__ == '__main__':
    with open('citscapes.json') as data_file:
        data = json.load(data_file)

    color_dict = {}
    palette = data['palette']
    for i, color in enumerate(palette):
        color_dict[tuple(color)] = i

    img_names = os.listdir(./semantics)

    for name in img_names:
        image = io.imread(os.path.join(self.root_dir, img_name))
        new_img = np.array((image.shape[1], image.shape[2]))
        for i in range(image.shape[1]):
            for j in range(image.shape[2]):
                label = color_dict[tuple(image[:, i, j])]
                new_img[i, j] = label

        np.save(new_imgs, './maps/' + name)



