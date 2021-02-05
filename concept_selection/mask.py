import torch
import numpy as np

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.long().cuda()
    return np_mask

def create_masks(src, trg):
    src_mask = (src != 0).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        np_mask = np_mask.cuda()
        trg_mask = trg_mask.cuda()
        np_mask = np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask


