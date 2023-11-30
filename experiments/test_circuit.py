import sys
sys.path.append('../')
import os
import torch
import numpy as np
from codebase import utils as ut
from codebase import metrics as mt
from utils import get_simplified_circuit_data
from torchvision.utils import save_image
from models.icm_vae import ICM_VAE


DATA="circuit"
SAVE_DIR="icm_vae_recon"
NAME="icm_vae_cdp_beta=0.05"
DATASET_DIR='../data/causal_circuit'
RUN=0
TRAIN=1
ITER_SAVE=5
device = torch.device("cuda:4" if(torch.cuda.is_available()) else "cpu")


layout = [
	('model={:s}',  str(NAME)),
	('run={:04d}', RUN),
	('toy={:s}', str(DATA) + '_' + str(NAME))
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
print('Model name:', model_name)



if not os.path.exists(f'./results/{DATA}/{DATA}_{NAME}_inference/'):
	os.makedirs(f'./results/{DATA}/{DATA}_{NAME}_inference/')
	

def save_model_by_name(model, global_step):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
	

C = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
scale = np.array([[0,1],[0,1],[0,1],[0,1]])



icm_vae = ICM_VAE(name=NAME + '_' + DATA, dataset="circuit", z_dim=16, z1_dim=4, z2_dim=4, C=C, scale=scale).to(device)
ut.load_model_by_name(icm_vae, 95)

train_dataset = get_simplified_circuit_data(DATASET_DIR, 64)
test_dataset = get_simplified_circuit_data(DATASET_DIR, 64, dataset="test")

optimizer = torch.optim.Adam(icm_vae.parameters(), lr=1e-3, betas=(0.9, 0.999))

icm_vae.eval()
rep_train = np.empty((35527, 16))
y_train = np.empty((35527, 4))

for batch_idx, (X, u) in enumerate(train_dataset):
    X = X.to(device)
    u = u.to(device)
    L, kl, rec, reconstructed_image, z, cp_m = icm_vae.forward(X,u,sample = False)
    z = z.reshape(-1, 16)
    rep_train[batch_idx*64:(batch_idx*64)+z.shape[0], :] = z.cpu().detach().numpy()
    y_train[batch_idx*64:(batch_idx*64)+u.shape[0], :] = u.cpu().detach().numpy()


icm_vae.eval()
total_loss = 0
total_rec = 0
total_kl = 0
rep_test = np.empty((3608, 16))
y_test = np.empty((3608, 4))

for batch_idx, (X, u) in enumerate(test_dataset):
    X = X.to(device)
    u = u.to(device)
    L, kl, rec, reconstructed_image, z, cp_m = icm_vae.forward(X,u,sample = False)
    z = z.reshape(-1, 16)
    rep_test[batch_idx*64:(batch_idx*64)+z.shape[0], :] = z.cpu().detach().numpy()
    y_test[batch_idx*64:(batch_idx*64)+u.shape[0], :] = u.cpu().detach().numpy()

    m = len(test_dataset)
    save_image(X, f'./results/{DATA}/{DATA}_{NAME}_inference/true.png')
    save_image(reconstructed_image, f'./results/{DATA}/{DATA}_{NAME}_inference/reconstructed.png')
	

scores, importance_matrix, code_importance = mt._compute_dci(rep_train.T, y_train.T, rep_test.T, y_test.T)
irs_score = mt.compute_irs(rep_train.T, y_train.T)

print(f'DCI Scores: {scores}')
print(f'Importances: {importance_matrix}')
print(f'IRS Score: {irs_score}')