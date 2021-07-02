import argparse
import time
import torch
import torch.nn as nn
from model import get_model
import torch.nn.functional as F
from mask import create_masks
import dill as pickle
import itertools
from tqdm import tqdm
from dataset import get_loader
import math
from build_vocab import Vocabulary
import numpy
import sys
from torch.optim.lr_scheduler import StepLR
numpy.set_printoptions(threshold=sys.maxsize)
torch.cuda.set_device(0)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def random_seeding(seed_value=1202, use_cuda=True):
    numpy.random.seed(seed_value) 
    torch.manual_seed(seed_value) 
    if use_cuda: torch.cuda.manual_seed_all(seed_value) # gpu vars


def train_model(transformer, dataset, epochs, criterion, optimizer):
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    print("training model...")
    for epoch in range(epochs):

        transformer.train()
        cur_lr = get_lr(optimizer)
        print("Current lr ", cur_lr)
        total_loss = []
        for index, (im_feats, word_idx, adjacent_matrix, trg_matrix) in enumerate(tqdm(dataset)):
            # B * S
            im_feats = im_feats.cuda()
            word_idx = word_idx.cuda()
            adjacent_matrix = adjacent_matrix.cuda()
            trg_matrix = trg_matrix.cuda()
            src_mask = None
            preds, _ = transformer(im_feats, word_idx, adjacent_matrix, src_mask)
            # print(preds.size(), trg_matrix.size())
            loss = criterion(preds.view(-1), trg_matrix.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.01)
            optimizer.step()
            total_loss.append(loss.item())

        scheduler.step()
        print(f"Epoch {epoch} training loss : ", sum(total_loss)/len(total_loss))
        if epoch % 10 == 0 and epoch >= 10:
            torch.save(transformer.state_dict(), f'models/transformer{epoch}.pth')

def main():
    # random_seeding()
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', type=str, default='../data/concept_selection/train.pkl')
    parser.add_argument('-test_src_data', type=str, default='../data/concept_selection/test.pkl')

    parser.add_argument('-epochs', type=int, default=101)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=64)
    parser.add_argument('-lr', type=int, default=0.0004)
    parser.add_argument('-max_strlen', type=int, default=700)

    opt = parser.parse_args()
    
    with open('vocab/vocab.pkl','rb') as f:
        vocab = pickle.load(f)
    # get model
    transformer = get_model(len(vocab), None, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("transformer trainable parameters: ", total_params)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    dataset = get_loader(opt.src_data, vocab, opt.batchsize, train=True)
    train_model(transformer, dataset, opt.epochs, criterion, optimizer)


if __name__ == "__main__":
    main()
