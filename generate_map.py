import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import IPython

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 512)
    return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(393216, 1000)
        self.fc21 = nn.Linear(1000, 100)
        self.fc22 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1000)
        self.fc4 = nn.Linear(1000, 393216)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == '__main__':
    model = VAE()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load('./vae.pth'))

    # generate random encoding
    encoding = Variable(torch.randn(1, 100))
    if torch.cuda.is_available():
        encoding = encoding.cuda()
    
    # decode 
    map = model.decode(encoding)

    # form image
    img = to_img(map.cpu().data)
    save_image(img, "test_decode.png")
