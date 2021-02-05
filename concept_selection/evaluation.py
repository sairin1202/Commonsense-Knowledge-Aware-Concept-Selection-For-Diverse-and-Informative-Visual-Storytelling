import math
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from mask import create_masks, nopeak_mask
import pickle
import argparse
from model import get_model
from nltk.corpus import wordnet
from build_vocab import Vocabulary
from copy import deepcopy
from dataset import get_loader
import matplotlib.pyplot as plt
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from bronkerbosch import bronker_bosch2
from bronkerbosch import Reporter
from nltk import ngrams

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


def score(hypo, reference, score_dic):
    max_bleu = -1
    max_rouge = -1
    print(hypo)
    print(reference)
    for ref in reference:
        bleu = 0
        rouge = 0
        n=1
        hypo_n = hypo
        ref_n = ref
        print('hypo', hypo_n)
        print('ref', ref_n)

        for h in hypo_n:
            if h in ref_n:
                bleu += 1
        for r in ref_n:
            if r in hypo_n:
                rouge += 1
        if rouge > max_rouge:
            max_bleu = bleu
            max_rouge = rouge
            len_hypo_n = len(hypo_n)
            len_ref_n = len(ref_n)
    print(max_bleu, max_rouge, len(hypo_n), len(ref_n))
    score_dic[f'bleu1_match'] += max_bleu
    score_dic[f'rouge1_match'] += max_rouge
    score_dic[f'bleu1_total'] += len(hypo_n)
    score_dic[f'rouge1_total'] += len(ref_n)
    return score_dic



def metric(pred_story):
    score_dic = {'rouge1_match':0, 'rouge1_total':0, 'bleu1_match':0, 'bleu1_total':0, 'rouge2_match':0, 'rouge2_total':0, 'bleu2_match':0, 'bleu2_total':0}
    with open(f'data/test.pkl','rb') as f:
        res_story = pickle.load(f)

    for story_id , value in res_story.items():

        targets = value['target']
        pred_keys = pred_story[story_id]

        tgts = []
        # print(targets)
        for target in targets:
            assert len(target) == 5
            tgt = []
            for i, tok in enumerate(target):
                # print(tok)
                tgt += list(set(tok))
                tgt = list(set(tgt))
            tgts.append(tgt)


        preds = []
        for pred in pred_keys:
            preds += list(set(pred))
        preds = list(set(preds))

        # print('pred', preds)
        # print('tgt', tgts)

        score_dic = score(preds, tgts, score_dic)
        print(score_dic)
    print('bleu1:', score_dic['bleu1_match']/score_dic['bleu1_total']*100, 'rouge1:', score_dic['rouge1_match']/score_dic['rouge1_total']*100)

def get_clique_score(clique, mat):
    emission_score = 0
    transition_score = 0
    for node in clique:
        emission_score += math.log(mat[500][node])
    for node1 in clique:
        for node2 in clique:
            if node1 == node2:
                continue
            transition_score += math.log(mat[node1][node2])
    emission_score = emission_score/len(clique)
    transition_score = transition_score/((len(clique)-1)**2)
    return emission_score + transition_score

def generate_seq_from_mat(mat):
    # create adjacent matrix
    NEIGHBORS = [[]]
    for _ in range(500):
        NEIGHBORS.append([])
    # index 0 means nothing
    # print(mat)
    cliques = []
    threshold = torch.max(mat).item()*0.9
    while len(cliques) == 0:
        threshold *= 0.8
        for i in range(500):
            if mat[500][i] < threshold:
                continue
            for j in range(i+1, 500):
                if mat[i][j] > threshold or mat[j][i] > threshold:
                    NEIGHBORS[i+1].append(j+1)
                    NEIGHBORS[j+1].append(i+1)
        # print(NEIGHBORS)
        NODES = set(range(1, len(NEIGHBORS)))
        report = Reporter()
        bronker_bosch2([], set(NODES), set(), report, NEIGHBORS)
        cliques = report.get_cliques()
        # print(len(cliques))
    cliques = [[c-1 for c in cl] for cl in cliques]
    scores = []
    # print(cliques)
    for clique in cliques:
        scores.append(get_clique_score(clique, mat))
    # print(scores)
    idx = np.argsort(scores)[::-1]
    cliques = np.array(cliques)
    # print(idx)
    cliques = cliques[idx]
    # print(cliques)
    return cliques[0]

def get_target(hypo, concept_idxs, vocab):
    tokens = [[],[],[],[],[]]
    for tok in hypo:
        word = vocab.idx2word[concept_idxs[tok]]
        if tok < 100:
            tokens[0].append(word)
        if 100 <= tok < 200:
            tokens[1].append(word) 
        if 200 <= tok < 300:
            tokens[2].append(word) 
        if 300 <= tok < 400:
            tokens[3].append(word) 
        if 400 <= tok < 500:
            tokens[4].append(word)
    return tokens

def generate(transformer, vocab):
    transformer.eval()
    res = {}
    with open('data/test.pkl', 'rb') as f:
        test_story = pickle.load(f)


    for index, src in tqdm(test_story.items()):
        images = src['image']
        targets = src['target']
        concepts = src['good_concept_flatten']
        # print({i+1:concept for i, concept in enumerate(concepts)})
        # print(targets)
        im_feats = []
        for im in images:
            f = np.load(f'../../../AREL/dataset/resnet_features/fc/test/{im}.npy')
            im_feats.append(f)             
        # im_feats = self.feats[i]     
        im_feats = torch.tensor(im_feats).unsqueeze(0).cuda()

        concept_idxs = []
        for con in concepts:
            concept_idxs.append(vocab(con))

        concepts = torch.LongTensor(concept_idxs).unsqueeze(0).cuda()

        adjacent_matrix = get_matrix().unsqueeze(0).cuda()
        mat, _ = transformer(im_feats, concepts, adjacent_matrix, None)
        hypo = generate_seq_from_mat(mat.squeeze())
        token = get_target(hypo, concept_idxs, vocab)
        res[index] = token
        print(token)
    with open("res/edge2cluster.pkl", "wb") as f:
        pickle.dump(res, f)
    metric(res)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-beam_size', type=int, default=4)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-src_lang', default='en')
    parser.add_argument('-trg_lang', default='en')
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-load_epoch', type=int, default=10)

    opt = parser.parse_args()
    assert opt.beam_size > 0
    assert opt.max_len > 10

    # load vocab
    with open("vocab/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    # opt.load_epoch = None
    for load_epoch in [10]:
        transformer = get_model(len(vocab), load_epoch, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
        print('epoch', load_epoch)
        generate(transformer, vocab)


    # get_attn(transformer, eventer, pool, pe, test_dataset)

if __name__ == '__main__':
    main()


