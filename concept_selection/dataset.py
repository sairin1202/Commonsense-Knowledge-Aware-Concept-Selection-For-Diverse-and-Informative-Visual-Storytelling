import numpy as np
import os
from torch.utils.data import Dataset
import glob
import numpy as np
from tqdm import tqdm
import dill as pickle
import torch
from copy import deepcopy


def get_concept_id(concept, i, word):
    for j in range(i*100, (i+1)*100):
        if word == concept[j]:
            return j
    print(word, concept[i*100:(i+1)*100])
    return -1
        

def get_matrix():
    adjacent_matrix = np.ones((500,500),dtype=np.float)
    adjacent_matrix = torch.LongTensor(adjacent_matrix)
    
    adjacent_matrix = torch.cat([adjacent_matrix, torch.ones(500, 5).long()], dim=1)
    adjacent_matrix = torch.cat([adjacent_matrix, torch.ones(5, 505).long()], dim=0)
    # img attend own word
    # 1
    adjacent_matrix[500,100:] = 0
    adjacent_matrix[100:,500] = 0   
    # 2     
    adjacent_matrix[501,200:] = 0
    adjacent_matrix[200:,501] = 0    
    adjacent_matrix[501,:100] = 0
    adjacent_matrix[:100,501] = 0  
    # 3
    adjacent_matrix[502,:200] = 0
    adjacent_matrix[:200,502] = 0        
    adjacent_matrix[502,300:] = 0
    adjacent_matrix[300:,502] = 0   
    # 4     
    adjacent_matrix[503,:300] = 0
    adjacent_matrix[:300,503] = 0    
    adjacent_matrix[503,400:] = 0
    adjacent_matrix[400:,503] = 0        
    # 5    
    adjacent_matrix[504,:400] = 0
    adjacent_matrix[:400,504] = 0    
    adjacent_matrix[504,500:] = 0
    adjacent_matrix[500:,504] = 0          
    return adjacent_matrix

class ROCdataset():
    def __init__(self, src_dir, vocab, train=True):
        print("Loading data ...  ")
        # loading training data
        with open(src_dir, 'rb') as f:
            self.src = pickle.load(f)
            
        print('src data length', len(self.src))
        self.data_len = len(self.src)

        print("Loading vocab ... ")
        # loading vocab data
        self.vocab = vocab
        self.train = train
        self.keys = list(self.src.keys())
        # self.feats = []
        # for key in tqdm(self.keys):
        #     feats = []
        #     images = self.src[key]['image']
        #     for im in images:
        #         if self.train:
        #             if not os.path.exists(f'../../AREL/dataset/resnet_features/fc/train/{im}.npy'):
        #                 print('train', im)
        #         else:
        #             if not os.path.exists(f'../../AREL/dataset/resnet_features/fc/test/{im}.npy'):
        #                 print('test', im)
        #     self.feats.append(feats)

    def __getitem__(self, index):
 
        index = self.keys[index]
        src_idx = []
        src = self.src[index]
        images = self.src[index]['image']
        targets = self.src[index]['target']
        concepts = self.src[index]['concept_flatten']

        # print({i+1:concept for i, concept in enumerate(concepts)})
        # print(targets)
        im_feats = []
        for im in images:
            if self.train:
                f = np.load(f'../../../AREL/dataset/resnet_features/fc/train/{im}.npy')
                im_feats.append(f)
            else:
                f = np.load(f'../../../AREL/dataset/resnet_features/fc/test/{im}.npy')
                im_feats.append(f)             
        # im_feats = self.feats[i]     
        im_feats = torch.tensor(im_feats)   

        concept_idxs = []
        for con in concepts:
            concept_idxs.append(self.vocab(con))
        concepts = torch.LongTensor(concept_idxs)

        adjacent_matrix = get_matrix()


        targets = src['target']

        trg_idxs = []
        for i in range(5):
            trg = targets[i]
            for tok in trg:
                tok_id = get_concept_id(concepts, i, self.vocab(tok))
                assert tok_id != -1
                trg_idxs.append(tok_id)
        
        target_matrix = torch.zeros(501,501).float()
        for k in range(len(trg_idxs)):
            target_matrix[500][trg_idxs[k]] = 1
            for l in range(k+1, len(trg_idxs)):
                target_matrix[trg_idxs[k]][trg_idxs[l]] = 1
                target_matrix[trg_idxs[l]][trg_idxs[k]] = 1
    
        return im_feats, concepts, adjacent_matrix, target_matrix

    def __len__(self):
        return self.data_len

def get_loader(src_dir, vocab, batch_size, train,shuffle=True):
    dataset = ROCdataset(src_dir, vocab, train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=32,
                                              drop_last=True)
    return data_loader