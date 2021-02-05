import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle
import argparse
from nltk.corpus import wordnet
from build_vocab import Vocabulary
from dataset import get_loader
from copy import deepcopy
import opts
import misc.utils as utils
from vist_eval.album_eval import AlbumEvaluator
import json
from bart import BART
total = 0
correct = 0
class Evaluator:
    def __init__(self, opt):
        ref_json_path = "data/reference/test_reference.json"
        self.reference = json.load(open(ref_json_path))
        print("loading file {}".format(ref_json_path))
        self.eval = AlbumEvaluator()

    def measure(self, filename):
        self.prediction_file = filename
        predictions = {}
        with open(self.prediction_file) as f:
            for line in f:
                vid, seq = line.strip().split('\t')
                if vid not in predictions:
                    predictions[vid] = [seq]
        self.eval.evaluate(self.reference, predictions)
        with open(filename, 'w') as f:
            json.dump(predictions, f)
        return self.eval.eval_overall


# def generate_sentence(album, src, model):
#     images = src
#     feats = []
#     for im in images:
#         feat = np.load(f'../../../AREL/dataset/resnet_features/fc/test/{im}.npy')
#         feats.append(deepcopy(feat))
#     feats = torch.tensor(feats).unsqueeze(0)
#     story = model.generate(cond=feats, top_k=-1, top_p=0.9)
#     print(album, story)
#     return story

# def generate(opt, model):
#     evaluator = Evaluator(opt)
#     with open('data/new_test_story.pkl', 'rb') as f:
#         test_src = pickle.load(f)
#     keys = list(test_src.keys())
#     # keys = [keys[i] for i in range(0,1000,5)]
#     hypos = {}
#     images = []
#     res = []
#     for album_id in tqdm(keys):
#         src = test_src[album_id]
#         hypo = generate_sentence(album_id, src, model)
#         hypos[album_id] = hypo
#         res.append(f'{album_id}\t {hypo}'+'\n')

#     with open("res/hypo.txt", "w") as f:
#         f.writelines(res)
#     evaluator.measure("res/hypo.txt")




def generate_sentence(album, src, model):
    with open('data/new_test_story.pkl','rb') as f:
        test_src = pickle.load(f)
    
    images = test_src[album]
    feats = []
    for im in images:
        feat = np.load(f'../../../AREL/dataset/resnet_features/fc/test/{im}.npy')
        feats.append(deepcopy(feat))
    feats = torch.tensor(feats).unsqueeze(0)

    keywords = src
    # keywords = ' '.join(keywords)
    story = model.generate(cond=feats, keys=keywords, top_k=-1, top_p=0.9)
    print(album, story)
    return story

def generate(opt, model):
    global total
    global correct
    evaluator = Evaluator(opt)

    with open('res/graph2keyword.json','r') as f:
        test_key = json.load(f)  
    album_ids = list(test_key.keys())
    with open('data/album2story.json','r') as f:
        album2story = json.load(f)
    story2album = {v: k for k, v in album2story.items()}

    # keys = [keys[i] for i in range(0,1000,5)]
    hypos = {}
    images = []
    res = []

    with open('data/new_test_story.pkl','rb') as f:
        test_src = pickle.load(f)
    
# it was a cold winter day . the birds were out in the lake . the birds were trying to fly . they spent a lot of time in the water . they could n't get back up 

    for album_id in tqdm(album_ids):
        src = test_key[album_id]

        image_keywords = []
        images = test_src[album_id]
        for image in images:
            with open(f'../../data/clarifai/test/{image}.json','rb') as f:
                image_key = json.load(f)
            image_keywords.append(image_key)

        keywords = []
        print(src)
        keywords = " <SEP> "
        for l, keys in enumerate(src):
            keys = keys + image_keywords[l]
            for k in keys:
                keywords += k+' '
            keywords += '<SEP> '
        keywords = keywords[:-1]
        hypo = generate_sentence(album_id, keywords, model)

        for keys in src:
            for k in keys:
                if k in hypo:
                    correct += 1
                total += 1
        print(correct/total)
        hypos[album_id] = hypo
        res.append(f'{album_id}\t {hypo}'+'\n')

    with open("res/bart.txt", "w") as f:
        f.writelines(res)
    evaluator.measure("res/bart.txt")





def main():
    global total
    global correct
    opt = opts.parse_opt()
    # load vocab
    # get model
    # opt.load_epoch = 3
    bart = BART(opt)
    bart.load_model(f'models/3_model.pt')
    # generate(opt, model, SRC, TRG, opt.beam_size)
    generate(opt, bart)
    print(correct/total)

if __name__ == '__main__':
    main()
