from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from .bleu import Bleu
import argparse
import codecs
import numpy as np
import math
from bert_score import score
from rouge import Rouge
import os, re
import ipdb
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity as cosine

def cal_aggregate_BLEU_nltk(refs, tgts):
    #print(refs)
    #print(tgts)
    smoothie = SmoothingFunction().method7
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    refs = [[ref.split(' ')] for ref in refs]
    tgts = [tgt.split(' ') for tgt in tgts]
    result = []
    for i in range(0,4):
        result = result + [corpus_bleu(refs, tgts, weights=weights[i], smoothing_function=smoothie)]
    return result[0], result[1], result[2], result[3]

# BLEU of NLTK
def cal_BLEU_nltk(refer, candidate, ngram=1):
    '''
    SmoothingFunction refer to https://github.com/PaddlePaddle/models/blob/a72760dff8574fe2cb8b803e01b44624db3f3eff/PaddleNLP/Research/IJCAI2019-MMPMS/mmpms/utils/metrics.py
    '''
    smoothie = SmoothingFunction().method7
    if ngram == 1:
        weight = (1, 0, 0, 0)
    elif ngram == 2:
        weight = (0.5, 0.5, 0, 0)
    elif ngram == 3:
        weight = (0.33, 0.33, 0.33, 0)
    elif ngram == 4:
        weight = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu(refer, candidate, 
                         weights=weight, 
                         smoothing_function=smoothie)

# BLEU of nlg-eval
def cal_BLEU(refs, tgts):
    scorer = Bleu(4)
    refs = {idx: [line] for idx, line in enumerate(refs)}
    tgts = {idx: [line] for idx, line in enumerate(tgts)}
    s = scorer.compute_score(refs, tgts)
    return s[0]

# BLEU of multibleu.perl
def cal_BLEU_perl(dataset, model):
    p = os.popen(f'python ./metric/perl-bleu.py {dataset} {model}').read()
    print(f'[!] multi-perl: {p}')
    pattern = re.compile(r'(\w+\.\w+)/(\w+\.\w+)/(\w+\.\w+)/(\w+\.\w+)')
    bleu1, bleu2, bleu3, bleu4 = pattern.findall(p)[0]
    bleu1, bleu2, bleu3, bleu4 = float(bleu1), float(bleu2), float(bleu3), float(bleu4)
    return bleu1, bleu2, bleu3, bleu4


def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity

def cal_intra_Distinct(lines):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    unary = []
    binary = []
    
    for line in lines:
      uni_distinct, bi_distinct = cal_Distinct(line)
      unary.append(uni_distinct)
      
      binary.append(bi_distinct)

    return float(np.mean(unary)), float(np.mean(binary))


embed = nn.Embedding(18091, 300)

def init_embed(vocab_size, embed_size):
    embed = nn.Embedding(vocab_size, embed_size)
    
def embedding(seqs):
    return embed(torch.tensor(seqs)).detach().numpy()

def mean(embs, lens):
    batch_size, seq_len, emb_size=embs.shape
    mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
    for i,length in enumerate(lens):
        mask[i,:length,:]=1
    return (embs*mask).sum(1)/(mask.sum(1)+1e-8)

    

def sim_bow(pred, pred_lens, ref, ref_lens):
    """
    :param pred - ndarray [batch_size x seqlen]
    :param pred_lens - list of integers
    :param ref - ndarray [batch_size x seqlen]
    """
    # look up word embeddings for prediction and reference
    emb_pred = embedding(pred) # [batch_sz x seqlen1 x emb_sz]
    emb_ref = embedding(ref) # [batch_sz x seqlen2 x emb_sz]

    ext_emb_pred=extrema(emb_pred, pred_lens)
    ext_emb_ref=extrema(emb_ref, ref_lens)
    bow_extrema=cosine(ext_emb_pred, ext_emb_ref) # [batch_sz_pred x batch_sz_ref]

    avg_emb_pred = mean(emb_pred, pred_lens) # Calculate mean over seq
    avg_emb_ref = mean(emb_ref, ref_lens) 
    bow_avg = cosine(avg_emb_pred, avg_emb_ref) # [batch_sz_pred x batch_sz_ref]


    batch_pred, seqlen_pred, emb_size=emb_pred.shape
    batch_ref, seqlen_ref, emb_size=emb_ref.shape
    cos_sim = cosine(emb_pred.reshape((-1, emb_size)), emb_ref.reshape((-1, emb_size))) # [(batch_sz*seqlen1)x(batch_sz*seqlen2)]
    cos_sim = cos_sim.reshape((batch_pred, seqlen_pred, batch_ref, seqlen_ref))
    # Find words with max cosine similarity
    max12 = cos_sim.max(1).mean(2) # max over seqlen_pred
    max21 = cos_sim.max(3).mean(1) # max over seqlen_ref
    bow_greedy=(max12+max21)/2 # [batch_pred x batch_ref(1)]
    return np.max(bow_extrema), np.max(bow_avg), np.max(bow_greedy)

def extrema(embs, lens): # embs: [batch_size x seq_len x emb_size]  lens: [batch_size]
    """
    computes the value of every single dimension in the word vectors which has the greatest
    difference from zero.
    :param seq: sequence
    :param seqlen: length of sequence
    """
    # Find minimum and maximum value for every dimension in predictions
    batch_size, seq_len, emb_size = embs.shape
    max_mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
    for i,length in enumerate(lens):
        max_mask[i,:length,:]=1
    min_mask = 1-max_mask
    seq_max = (embs*max_mask).max(1) # [batch_sz x emb_sz]
    seq_min = (embs+min_mask).min(1)
    # Find the maximum absolute value in min and max data
    comp_mask = seq_max >= np.abs(seq_min)# [batch_sz x emb_sz]
    # Add vectors for finding final sequence representation for predictions
    extrema_emb = seq_max* comp_mask + seq_min* np.logical_not(comp_mask)
    return extrema_emb


def cal_ROUGE(refer, candidate):
    if len(candidate) == 0:
        candidate = ['<unk>']
    elif len(candidate) == 1:
        candidate.append('<unk>')
    if len(refer) == 0:
        refer = ['<unk>']
    elif len(refer) == 1:
        refer.append('<unk>')
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(candidate), ' '.join(refer))
    return scores[0]['rouge-2']['f']


def cal_BERTScore(refer, candidate):
    # too slow, fuck it
    _, _, bert_scores = score(candidate, refer, lang='en', rescale_with_baseline=True)
    bert_scores = bert_scores.tolist()
    bert_scores = [0.5 if math.isnan(score) else score for score in bert_scores]
    return np.mean(bert_scores)

# ========== fuck nlg-eval fuck ========== #
# ========== Our own embedding-based metric ========== #
def cal_vector_extrema(x, y, dic):
    # x and y are the list of the words
    # dic is the gensim model which holds 300 the google news word2ved model
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w.lower()])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    vec_x = np.max(x, axis=0)
    vec_y = np.max(y, axis=0)
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.zeros(len(vec_x))
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def cal_embedding_average(x, y, dic):
    # x and y are the list of the words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w.lower()])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    
    vec_x = np.array([0 for _ in range(len(x[0]))])
    for x_v in x:
        x_v = np.array(x_v)
        vec_x = np.add(x_v, vec_x)
    vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
    
    vec_y = np.array([0 for _ in range(len(y[0]))])
    #print(len(vec_y))
    for y_v in y:
        y_v = np.array(y_v)
        vec_y = np.add(y_v, vec_y)
    vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
    
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    
    zero_list = np.array([0 for _ in range(len(vec_x))])
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    
    vec_x = np.mat(vec_x)
    vec_y = np.mat(vec_y)
    num = float(vec_x * vec_y.T)
    denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
    cos = num / denom
    
    # res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    
    return cos


def cal_greedy_matching(x, y, dic):
    # x and y are the list of words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w.lower()])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    
    len_x = len(x)
    len_y = len(y)
    
    cosine = []
    sum_x = 0 

    for x_v in x:
        for y_v in y:
            assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(x_v))

            if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                if x_v.all() == y_v.all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                # method 1
                res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)
        if cosine:
            sum_x += max(cosine)
            cosine = []

    sum_x = sum_x / len_x
    cosine = []

    sum_y = 0

    for y_v in y:

        for x_v in x:
            assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(y_v))

            if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                if (x_v == y_v).all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                # method 1
                res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)

        if cosine:
            sum_y += max(cosine)
            cosine = []

    sum_y = sum_y / len_y
    score = (sum_x + sum_y) / 2
    return score


def cal_greedy_matching_matrix(x, y, dic):
    # x and y are the list of words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w.lower()])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)     # [x, 300]
    y = vecterize(y)     # [y, 300]
    
    len_x = len(x)
    len_y = len(y)
    
    matrix = np.dot(x, y.T)    # [x, y]
    matrix = matrix / np.linalg.norm(x, axis=1, keepdims=True)    # [x, 1]
    matrix = matrix / np.linalg.norm(y, axis=1).reshape(1, -1)    # [1, y]
    
    x_matrix_max = np.mean(np.max(matrix, axis=1))    # [x]
    y_matrix_max = np.mean(np.max(matrix, axis=0))    # [y]
    
    return (x_matrix_max + y_matrix_max) / 2
    
    
    
    
# ========== End of our own embedding-based metric ========== #



if __name__ == "__main__":
    path = './processed/dailydialog/GatedGCN-no-correlation/pred.txt'
    with open(path) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            if idx % 4 == 1:
                line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())
                
    # Distinct-1, Distinct-2
    candidates, references = [], []
    for line1, line2 in zip(tgt, ref):
        candidates.extend(line1)
        references.extend(line2)
    distinct_1, distinct_2 = cal_Distinct(candidates)
    rdistinct_1, rdistinct_2 = cal_Distinct(references)
    
    print(distinct_1, distinct_2)
