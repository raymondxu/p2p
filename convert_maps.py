import numpy as np
import os
from skimage import io
import json

if __name__ == '__main__':
    with open('cityscapes.json') as data_file:
        data = json.load(data_file)

    color_dict = {}
    palette = data['palette']
    for i, color in enumerate(palette):
        color_dict[tuple(color)] = i

    img_names = os.listdir('./semantics')

    for name in img_names:
        image = io.imread(os.path.join('./semantics', name))
        new_img = np.empty((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                color = tuple(image[i, j, :])
                if color in color_dict:
                    label = color_dict[tuple(image[i, j, :])]
                else:
                    label = -1
                new_img[i, j] = label

        np.save('./maps/' + name, new_img)



