import numpy as np
import os
from torch.utils.data import Dataset
import glob
import numpy as np
from tqdm import tqdm
import dill as pickle
import torch
from copy import deepcopy
import json
import random
class ROCdataset():
    def __init__(self, src_dir, train=True):
        print("Loading data ...  ")
        # loading training data
        with open(src_dir, 'rb') as f:
            self.src = pickle.load(f)
            
        print('src data length', len(self.src))
        self.data_len = len(self.src)

        print("Loading vocab ... ")
        # loading vocab data
        self.train = train
        self.keys = list(self.src.keys())

    def __getitem__(self, index):
        i = index
        index = self.keys[index]
        src_idx = []
        src = self.src[index]
        images = self.src[index]['image']

        texts = self.src[index]['text']
        # print(" ".join(texts))


        keywords = []

        tokens = self.src[index]['target']


        image_keywords = []
        for image in images:
            with open(f'../data/clarifai/train/{image}.json','rb') as f:
                image_key = json.load(f)
            image_keywords.append(image_key)
        # print(tokens)
        all_keywords = []
        for k ,token in enumerate(tokens):
            new_token = []
            token = token + image_keywords[k]
            for t in token:
                if np.random.rand() > 0.2:
                    new_token.append(t)
            token = new_token
            random.shuffle(token)
            all_keywords.append(token)

        keywords = " <SEP> "
        for keys in all_keywords:
            for k in keys:
                keywords += k+' '
            keywords += '<SEP> '
        keywords = keywords[:-1]
        
        im_feats = []
        for im in images:
            if self.train:
                f = np.load(f'../data/AREL/dataset/resnet_features/fc/train/{im}.npy')
                im_feats.append(f)
            else:
                f = np.load(f'../data/AREL/dataset/resnet_features/fc/test/{im}.npy')
                im_feats.append(f)             
        # im_feats = self.feats[i]     
        im_feats = torch.tensor(im_feats)
        return im_feats, " "+" ".join(texts), keywords

    def __len__(self):
        return 10
        return self.data_len




def get_loader(src_dir, batch_size, train,shuffle=True):
    dataset = ROCdataset(src_dir, train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=10,
                                              drop_last=True)
    return data_loader
