from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from PIL import Image


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)

dataset = data_loader.get_dataset()
print(len(data_loader))

maps = dataset.get_resized()
print(len(maps))


print('saving')
DIR = 'resized_maps'

for i, m in enumerate(maps):
    m.save('{}/{}.png'.format(DIR, i + 1))
