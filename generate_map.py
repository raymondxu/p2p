import torch
import vae
import numpy as np
import IPython

if __name__ == '__main__':
	model = vae.VAE()
	model.load_state_dict(torch.load('./vae.pth'))

	# generate random encoding
	encoding = torch.Tensor(np.random.rand(100,))

	# decode 
	map = model.decode(encoding)

	IPython.embed()

