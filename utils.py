import io
import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset





def matrix_poly(matrix, d):
    x = torch.eye(d).to(matrix.device)+ torch.div(matrix, d)#.to(device)
    return torch.matrix_power(x, d)
    
    
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A
    

def mask_threshold(x):
  x = (x+0.5).int().float()
  return x


class CausalCircuit(Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset
        
        self.imgs = []
        self.labels = []
        
        if dataset == "train":
            for k in range(4):
                data = np.load(f'../../data/causal_data/causal_circuit/train-{k}.npz') # np.load('../../data/causal_data/causal_circuit/train-0.npz')
                self.img_labels = data['original_latents'][:, 0, :]
                
                
                indices_11 = np.argwhere((self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4))
                self.img_labels_1 = self.img_labels[(self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4)]
                self.img_labels = self.img_labels_1
                
                temp = data['imgs'][:, 0]
                filtered_images = np.take(temp, indices_11)
                for i in range(len(filtered_images)):
                    self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                    self.labels.append(self.img_labels[i])
        else:
            data = np.load('../../data/causal_data/causal_circuit/test.npz')
            self.img_labels = data['original_latents'][:, 0, :]
                
                
            indices_11 = np.argwhere((self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4))
            self.img_labels_1 = self.img_labels[(self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4)]
            self.img_labels = self.img_labels_1

            temp = data['imgs'][:, 0]
            filtered_images = np.take(temp, indices_11)
            for i in range(len(filtered_images)):
                self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                self.labels.append(self.img_labels[i])

        self.dataset = dataset
        self.transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        data = self.imgs[idx]
        # print(np.asarray(self.labels).reshape(35527, 4))
        perm = [3, 2, 1, 0]
        label = torch.from_numpy(np.asarray(self.labels)[idx][perm])


        if self.transforms:
            data = self.transforms(data)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)


    
class CausalCircuitSimplified(Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset
        
        self.imgs = []
        self.labels = []
        
        if dataset == "train":
            for k in range(10):
                data = np.load(f'../../data/causal_data/causal_circuit/train-{k}.npz')

                perm = [3, 2, 1, 0]
                self.img_labels_0 = data['original_latents'][:, 0, :]
                self.img_labels_1 = data['original_latents'][:, 1, :]
                # self.img_labels = data['original_latents'][:, 0, :][:, perm]
                # THREE CASES

                self.img_labels = np.concatenate((self.img_labels_0, self.img_labels_1))
                # print(self.img_labels.shape)

                indices_11 = np.argwhere((self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2))
                self.img_labels_1 = self.img_labels[(self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2)]

                indices_12 = np.argwhere((self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2))
                self.img_labels_2 = self.img_labels[(self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2)]

                indices_13 = np.argwhere((self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2))
                self.img_labels_3 = self.img_labels[(self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2)]

                # indices_14 = np.argwhere((self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1))
                # self.img_labels_4 = self.img_labels[(self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1)]


                self.img_labels = np.concatenate((self.img_labels_1, self.img_labels_2, self.img_labels_3))
                # print(self.img_labels.shape)


                indices = np.concatenate((indices_11, indices_12, indices_13))

                # print(self.img_labels.shape)

                temp1 = data['imgs'][:, 0]
                temp2 = data['imgs'][:, 1]

                temp = np.concatenate((temp1, temp2))
                #filtered_images = temp
                filtered_images = np.take(temp, indices)
                # print(filtered_images)

                for i in range(len(filtered_images)):
                    self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                    self.labels.append(self.img_labels[i])

        else:
            data = np.load('../../data/causal_data/causal_circuit/test.npz')
            self.imgs = []
        
            perm = [3, 2, 1, 0]
            self.img_labels_0 = data['original_latents'][:, 0, :]
            self.img_labels_1 = data['original_latents'][:, 1, :]
            # self.img_labels = data['original_latents'][:, 0, :][:, perm]
            # THREE CASES

            self.img_labels = np.concatenate((self.img_labels_0, self.img_labels_1))
            print(self.img_labels.shape)

            indices_11 = np.argwhere((self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2))
            self.img_labels_1 = self.img_labels[(self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2)]

            indices_12 = np.argwhere((self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2))
            self.img_labels_2 = self.img_labels[(self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2)]

            indices_13 = np.argwhere((self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2))
            self.img_labels_3 = self.img_labels[(self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2)]

    #         indices_14 = np.argwhere((self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1))
    #         self.img_labels_4 = self.img_labels[(self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1)]


            self.labels = np.concatenate((self.img_labels_1, self.img_labels_2, self.img_labels_3))       
            indices = np.concatenate((indices_11, indices_12, indices_13))


            temp1 = data['imgs'][:, 0]
            temp2 = data['imgs'][:, 1]

            temp = np.concatenate((temp1, temp2))
            filtered_images = np.take(temp, indices)

            for i in range(len(filtered_images)):
                self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                    
        self.dataset = dataset
        self.transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        data = self.imgs[idx]
        # print(np.asarray(self.labels).reshape(35527, 4))
        perm = [3, 2, 1, 0]
        label = torch.from_numpy(np.asarray(self.labels)[idx][perm])

        if self.transforms:
            data = self.transforms(data)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)
    
    


class SyntheticLabeled(data.Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset

        imgs = os.listdir(root)

        self.dataset = dataset
        
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.imglabel = [list(map(int,k[:-4].split("_")[1:]))  for k in imgs]
        
        label = np.asarray(self.imglabel)
        self.label_avg = np.mean(label, axis=0)
        self.label_std = np.std(label, axis=0) 
        
        
        #print(self.imglabel)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]
        
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        array1 = np.asarray(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)
        
        
        
        

def tupleize_data(images):
    temp = images
    tup = []
    for i in range(len(temp)):
        lst = temp[i]
        # print(lst)

        for idx, item in enumerate(lst):
            # print(str(item))
            # print(str(item)[-10:])
            if 'orig' in item[-10:]:
                img = images[i].pop(idx)
                # print(img)

        for item in images[i]:
            tup.append((img, item))

    return tup


class SyntheticPaired(Dataset):
    def __init__(self, root, dataset="train", mode="train"):
        root = root + "/" + dataset
        self.mode = mode
        imgs = os.listdir(root)

        imgs = [os.path.join(root, k) for k in imgs]

        final_img = []
        for img in imgs:
            ims = os.listdir(img)
            ims = [os.path.join(img, k) for k in ims]
            final_img.append(ims)

        self.data = tupleize_data(final_img)

        # Labels of pre and post intervention samples
        # self.u_x = [list(map(int, k[0].split('/')[6].strip('.png').split('_')[1:5])) for k in self.data]
        # self.u_x = torch.tensor(self.u_x, dtype=torch.float32)
        self.u_x = [list(map(int, k[0].split('/')[-1].strip('.png').split('_')[-5:-1])) for k in self.data]
        self.u_x = torch.tensor(self.u_x, dtype=torch.float32)

        
        # for i in range(u_x.shape[0]):
        #     u_x[i] = (u_x[i].to(device) - torch.tensor(scale[:, 0])) / (
        #             torch.tensor(scale[:, 1]))
        
        # scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        # self.u_x = (self.u_x - torch.tensor(scale[:, 0], dtype=torch.float32)) / (
        #     torch.tensor(scale[:, 1], dtype=torch.float32))
        
        # for k in self.data:
        #     print(k)

        # self.u_y = [list(map(int, k[1].split('/')[6].strip('.png').split('_')[1:5])) for k in self.data]
        # self.u_y = torch.tensor(self.u_y)
        self.u_y = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[-5:-1])) for k in self.data]
        self.u_y = torch.tensor(self.u_y, dtype=torch.float32)
        
        # self.u_y = (self.u_y - torch.tensor(scale[:, 0], dtype=torch.float32)) / (
        #     torch.tensor(scale[:, 1], dtype=torch.float32))
        
        # Intervention Target
        # self.I = [list(map(int, k[1][-7:-4].split('_')[1:])) for k in self.data]
        # self.I = [list(map(int, k[1].split('/')[6].strip('.png').split('_')[5])) for k in self.data]
        self.I = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[-1])) for k in self.data]


        self.I_one_hot = np.eye(4)[self.I].astype('float32')

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path_1 = self.data[idx][0]
        img_path_2 = self.data[idx][1]

        x = np.asarray(Image.open(img_path_1))
        x_int = np.asarray(Image.open(img_path_2))

        target = torch.from_numpy(np.asarray(self.I_one_hot[idx]))

        if self.transforms:
            x = self.transforms(x)
            x_int = self.transforms(x_int)
        else:
            x = np.from_numpy(np.asarray(x).reshape(96, 96, 4))
            x_int = np.from_numpy(np.asarray(x_int).reshape(96, 96, 4))

        return x, x_int, self.u_x[idx], self.u_y[idx], target

    def __len__(self):
        return len(self.data)

def get_paired_data2(dataset_dir, batch_size, dataset="train", mode="train"):
    dataset = SyntheticPaired(dataset_dir, dataset=dataset, mode=mode)
    
    train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * 0.7))))
    val_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.7), int(len(dataset) * 0.85))))
    test_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.85), len(dataset))))

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def get_paired_data(dataset_dir, batch_size, dataset="train", mode="train"):
    dataset = SyntheticPaired(dataset_dir, dataset=dataset, mode=mode)
    print(len(dataset))
    dataset_split = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7) + 1, int(len(dataset)*0.3)], generator=torch.Generator().manual_seed(42))
    train_dataset, test_dataset = dataset_split[0], dataset_split[1]
    
    val_split = torch.utils.data.random_split(test_dataset, [int(len(test_dataset)*0.5), int(len(test_dataset)*0.5)], generator=torch.Generator().manual_seed(42))
    val_dataset, test_dataset = val_split[0], val_split[1]
    
    # train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * 0.7))))
    # val_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.7), int(len(dataset) * 0.85))))
    # test_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.85), len(dataset))))

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader



def get_paired_loaders(train_dataset, val_dataset, test_dataset, batch_size):

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def get_batch_unin_dataset_withlabel(dataset_dir, batch_size, dataset="train"):
  
	dataset = SyntheticLabeled(dataset_dir, dataset)
	print(len(dataset))
	dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

	return dataloader



def get_circuit_data(dataset_dir, batch_size, dataset="train"):
    dataset = CausalCircuit(dataset_dir, dataset)
    print(len(dataset))
    
    dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)
    
    return dataloader


def get_simplified_circuit_data(dataset_dir, batch_size, dataset="train"):
    dataset = CausalCircuitSimplified(dataset_dir, dataset)
    print(len(dataset))
    
    dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)
    
    return dataloader



class dataload(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)