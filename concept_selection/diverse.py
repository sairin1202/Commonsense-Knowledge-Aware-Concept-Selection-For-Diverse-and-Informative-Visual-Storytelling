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


def score(hypo, reference, score_dic):
    for ref in reference:
        for n in range(1,2):
            hypo_n = list(ngrams(hypo, n))
            ref_n = list(ngrams(ref, n))
            # print(hypo_n)
            # print(ref_n)
            score_dic[f'bleu{n}_total'] += len(hypo_n)
            score_dic[f'rouge{n}_total'] += len(ref_n)
            for h in hypo_n:
                if h in ref_n:
                    score_dic[f'bleu{n}_match'] += 1
            for r in ref_n:
                if r in hypo_n:
                    score_dic[f'rouge{n}_match'] += 1
    return score_dic



def metric(res_story):
    score_dic = {'rouge1_match':0, 'rouge1_total':0, 'bleu1_match':0, 'bleu1_total':0, 'rouge2_match':0, 'rouge2_total':0, 'bleu2_match':0, 'bleu2_total':0}
    for story_id , value in res_story.items():
        with open(f'../graph_info/story/test/{story_id}.pkl','rb') as f:
            target = pickle.load(f)
        target_images = target['image']
        word2clusters = target['word2cluster']
        cluster2words = target['cluster2words']
        tok_clusters = target['token_cluster']
        tgt_clusters = []
        for i, tok_cluster in enumerate(tok_clusters):
            tgt_clusters += tok_cluster
        pred_clusters = value  
        pred_clusters = list(set(pred_clusters))
        # if len(pred_clusters) > 10:
        #     continue
        tgt_clusters = list(set(tgt_clusters))   
        # print('pred', pred_clusters)
        # print('tgt', tgt_clusters)
        score_dic = score(pred_clusters, [tgt_clusters], score_dic)
    print('bleu1:', score_dic['bleu1_match']/score_dic['bleu1_total']*100, 'rouge1:', score_dic['rouge1_match']/score_dic['rouge1_total']*100)



def get_clique_score(clique, mat):
    emission_score = 0
    transition_score = 0
    for node in clique:
        emission_score += math.log(mat[250][node])
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
    for _ in range(250):
        NEIGHBORS.append([])
    # index 0 means nothing
    # print(mat)
    cliques = []
    threshold = torch.max(mat).item()*0.9
    while len(cliques) < 20:
        threshold *= 0.8
        for i in range(250):
            if mat[250][i] < threshold:
                continue
            for j in range(i+1, 250):
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
    print(cliques[0],cliques[5],cliques[10], cliques[15], cliques[19])
    return [cliques[0],cliques[5],cliques[10], cliques[15], cliques[19]]


def generate(transformer, vocab, beam_size):
    transformer.eval()
    with open('data/keyword_test.json', 'rb') as f:
        test_story = json.load(f)

    with open('data/album2story.json','rb') as f:
        album2story = json.load(f)

    with open('vocab/unique.pkl','rb') as f:
        unique = pickle.load(f)

    sos_feat = unique['sos']
    eos_feat = unique['eos']


    story2album = {k:v for v, k in album2story.items()}

    res = [{},{},{},{},{}]
    res_story = {}
    hypos = []
    for idx, _ in tqdm(test_story.items()):
        if idx not in story2album:
            continue
        album_id = story2album[idx]
        with open(f'../graph_info/story/test/{idx}.pkl','rb') as f:
            value = pickle.load(f)

        cluster2feat = value['cluster2feat']

        word_feats = [cluster2feat[i] for i in range(250)]
        word_feats = torch.Tensor(word_feats)
        word_feats = torch.cat([word_feats, torch.Tensor(sos_feat).unsqueeze(0)], dim=-2)
        
        adjacent_matrix = value['adjacent_matrix']
        adjacent_matrix = torch.LongTensor(adjacent_matrix)
        # insert eos and image adjacent infomation
        adjacent_matrix = torch.cat([adjacent_matrix, torch.ones(250, 6).long()], dim=1)
        adjacent_matrix = torch.cat([adjacent_matrix, torch.ones(6, 256).long()], dim=0)     
        # sos attend everything
        # adjacent_matrix[250,:] = 0
        # adjacent_matrix[:,250] = 0
        # img attend own word
        # 1
        adjacent_matrix[251,50:] = 0
        adjacent_matrix[50:,251] = 0   
        # 2     
        adjacent_matrix[252,:50] = 0
        adjacent_matrix[:50,252] = 0    
        adjacent_matrix[252,100:] = 0
        adjacent_matrix[100:,252] = 0  
        # 3
        adjacent_matrix[253,:100] = 0
        adjacent_matrix[:100,253] = 0        
        adjacent_matrix[253,150:] = 0
        adjacent_matrix[150:,253] = 0   
        # 4     
        adjacent_matrix[254,:150] = 0
        adjacent_matrix[:150,254] = 0    
        adjacent_matrix[254,200:] = 0
        adjacent_matrix[200:,254] = 0        
        # 5    
        adjacent_matrix[255,:200] = 0
        adjacent_matrix[:200,255] = 0    
        adjacent_matrix[255,250:] = 0
        adjacent_matrix[250:,255] = 0      

        word_feats = word_feats.unsqueeze(0).cuda()
        adjacent_matrix = adjacent_matrix.unsqueeze(0).cuda()

        images = value['image']
        im_feats = []
        for im in images:
            f = np.load(f'../../../AREL/dataset/resnet_features/fc/test/{im}.npy')
            im_feats.append(f)

        im_feats = torch.Tensor(im_feats).unsqueeze(0).cuda()

        mat, _ = transformer(im_feats, word_feats, adjacent_matrix, None)
        hypo = generate_seq_from_mat(mat.squeeze())
        for x in range(5):
            res[x][album_id] = hypo[x]
        # print(hypo)
        # res_story[idx] = hypo
            with open(f"diverse/{x}.pkl", "wb") as f:
                pickle.dump(res[x], f)
    # metric(res_story)



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
    for load_epoch in [30]:
        transformer = get_model(251, load_epoch, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
        test_dataset = get_loader("data/new_keyword_test.json", 1, train=False)
        # evaluate
        # eval(transformer, test_dataset)

        # generate
        #  greedy_generate(model, SRC, TRG)
        print('epoch', load_epoch)
        generate(transformer, vocab, 4)


    # get_attn(transformer, eventer, pool, pe, test_dataset)

if __name__ == '__main__':
    main()


