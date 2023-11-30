import sys
sys.path.append('../')
import os
import torch
import numpy as np
from codebase import utils as ut
from utils import get_simplified_circuit_data
from torchvision.utils import save_image
from models.icm_vae import ICM_VAE


MAX_EPOCHS=100
DATA="circuit"
SAVE_DIR="icm_vae_recon"
NAME="icm_vae_cdp_beta=0.05"
DATASET_DIR='../data/causal_circuit'
RUN=0
TRAIN=1
ITER_SAVE=5
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")


layout = [
	('model={:s}',  str(NAME)),
	('run={:04d}', RUN),
	('toy={:s}', str(DATA) + '_' + str(NAME))
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
print('Model name:', model_name)


if not os.path.exists(f'./results/{DATA}/{DATA}_{NAME}_reconstructions/'):
	os.makedirs(f'./results/{DATA}/{DATA}_{NAME}_reconstructions/')
	


def save_model_by_name(model, global_step):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
	

C = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
scale = np.array([[0, 1],[0, 1],[0, 1],[0, 1]])

icm_vae = ICM_VAE(name=NAME + '_' + DATA, dataset="circuit", z_dim=128, z1_dim=4, z2_dim=32, C=C, scale=scale).to(device)

dataset_dir = '../../data/causal_data/causal_circuit'
train_loader = get_simplified_circuit_data(DATASET_DIR, 100)


optimizer = torch.optim.Adam(icm_vae.parameters(), lr=1e-3, betas=(0.9, 0.999))

def linear_scheduler(step, total_steps, initial, final):
    """Linear scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return (1.0 - t) * initial + t * final


for epoch in range(MAX_EPOCHS):
	icm_vae.train()
	total_loss = 0
	total_rec = 0
	total_kl = 0
	for X, l in train_loader:
		optimizer.zero_grad()
		#u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
		X = X.to(device)
		L, kl, rec, reconstructed_image, z, cp_m = icm_vae.forward(X,l,sample = False, lambdav=1e-5)
   
		L.backward()
		optimizer.step()

		total_loss += L.item()
		total_kl += kl.item() 
		total_rec += rec.item() 

		m = len(train_loader)
		save_image(X[0], f'./results/{DATA}/{DATA}_{NAME}_reconstructions/true_{epoch}.png')
		save_image(reconstructed_image[0], f'./results/{DATA}/{DATA}_{NAME}_reconstructions/reconstructed_{epoch}.png')

	beta = linear_scheduler(epoch, 94, 0.0, 0.05)
	icm_vae.beta = beta

	alpha = linear_scheduler(epoch, 94, 0.0, 0.1)
	icm_vae.alpha = alpha

	if epoch % 1 == 0:
		print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+'m:' + str(m))

	if epoch == 90 or epoch == 99 or epoch == 100:
		ut.save_model_by_name(icm_vae, epoch)