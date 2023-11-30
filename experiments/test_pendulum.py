import sys
sys.path.append('../')
import os
import torch
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from codebase import utils as ut
from codebase import metrics as mt
from utils import get_batch_unin_dataset_withlabel
from torchvision.utils import save_image
from models.icm_vae import ICM_VAE

DATA="pendulum"
SAVE_DIR="icm_vae_recon"
NAME="icm_vae_cdp"
DATASET_DIR='../data/pendulum'
RUN=0
TRAIN=1
ITER_SAVE=5
device = torch.device("cuda:5" if(torch.cuda.is_available()) else "cpu")

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


C = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
icm_vae = ICM_VAE(name=NAME + '_' + DATA, z_dim=16, z1_dim=4, z2_dim=4, C=C, scale=scale).to(device)
ut.load_model_by_name(icm_vae, 100)

train_dataset = get_batch_unin_dataset_withlabel(DATASET_DIR, 64, dataset="train")
test_dataset = get_batch_unin_dataset_withlabel(DATASET_DIR, 64, dataset="test")

icm_vae.eval()
rep_train = np.empty((5482, 16))
y_train = np.empty((5482, 4))
for batch_idx, (X, u) in enumerate(train_dataset):
    #u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
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
rep_test = np.empty((1826, 16))
y_test = np.empty((1826, 4))
for batch_idx, (X, u) in enumerate(test_dataset):
    #u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
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
























