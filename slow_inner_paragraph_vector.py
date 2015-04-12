#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import logging
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, mean as np_mean, sum as np_sum
import numpy
logger = logging.getLogger("gensim.models.word2vec")

import random

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange

# import pyximport; pyximport.install(pyimport = True)
# from word2vec_inner import train_sentence_sg, train_sentence_cbow, FAST_VERSION

# numpy.seterr(all='ignore')

import unicodedata
# import re
import zenhan

#random_seed
random_seed = 0.444298694171234


# 文字を正規化する
def clean_text(text):
    # del_n = re.compile('\n')
    # text = del_n.sub('',text)
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = zenhan.z2h(text,zenhan.ASCII|zenhan.DIGIT)
    return text

# 中間層はdoc_vecのみ
def train_sentence_sg_simple(model, sentence_id, sentence, alpha, work=None,alpha_doc=0.025):
    """
    Update skip-gram model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    # print sentence_id

    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.0

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window)  # `b` in the original word2vec code
        reduced_window = 0
        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                # l1 = model.syn0[word2.index]

                # print "l1.shape", l1.shape
                doc_vec = model.doc[sentence_id]
                # doc_vec_length = model.doc_vec_size

                l1 = doc_vec[:]
                # word_vec_length = l1.shape[0]
                
                neu1e = zeros(l1.shape)

                if model.hs:
                    # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                    # print "l2a.shape",l2a.shape
                    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                    ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                    model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
                    neu1e += dot(ga, l2a) # save error

                if model.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[numpy.random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error

  
                model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)

    return len([word for word in sentence if word is not None])



# 中間層はdoc_vecとword_vecの平均
def train_sentence_sg_average(model, sentence_id, sentence, alpha, work,neu1,alpha_doc):
    """
    Update skip-gram model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    # print sentence_id


    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.0

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window)  # `b` in the original word2vec code
        reduced_window = 0
        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                l1 = model.syn0[word2.index]

                # print "l1.shape", l1.shape
                doc_vec = model.doc[sentence_id]
                doc_vec_length = model.doc_vec_size

                l1 = (l1+doc_vec)/2

                neu1e = zeros((doc_vec_length),dtype=REAL)

                # word_vec_length = l1.shape[0]
                
                # neu1e = zeros(l1.shape)

                if model.hs:
                    # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                    # print "l2a.shape",l2a.shape
                    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                    ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                    model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
                    neu1e += dot(ga, l2a) # save error

                if model.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[numpy.random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error



                if not model.freeze_learn:
                    model.syn0[word2.index] += neu1e
                model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)

                # model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)

    return len([word for word in sentence if word is not None])



# 中間層はword_vecの連結
def train_sentence_sg_concat(model, sentence_id, sentence, alpha, work,neu1,alpha_doc):
    """
    Update skip-gram model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    # print sentence_id


    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.0

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window)  # `b` in the original word2vec code
        reduced_window = 0
        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                l1 = model.syn0[word2.index]

                word_vec_length = l1.shape[0]
                # print "l1.shape", l1.shape
                doc_vec = model.doc[sentence_id]
                doc_vec_length = model.doc_vec_size

                l1 = numpy.append(l1,doc_vec)

                neu1e = zeros((word_vec_length+doc_vec_length),dtype=REAL)
                # word_vec_length = l1.shape[0]
                
                # neu1e = zeros(l1.shape)

                if model.hs:
                    # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                    ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                    model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output

                    # print "l2a.shape",l2a.shape
                    # print "ga.shape",ga.shape
                    # print "dot(ga, l2a) ",dot(ga, l2a).shape
                    # print "neu1e", neu1e.shape
                    neu1e += dot(ga, l2a) # save error

                if model.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[numpy.random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error

                if not model.freeze_learn:
                    model.syn0[word2.index] += neu1e[:word_vec_length] # learn input -> hidden, here for all words in the window separately
                model.doc[sentence_id]    += neu1e[word_vec_length:]*(alpha_doc/alpha)

                # model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)

    return len([word for word in sentence if word is not None])



# # 中間層はword_vec,doc_vecの連結
# def train_sentence_sg(model, sentence_id, sentence, alpha, work=None,alpha_doc=0.025):
#     """
#     Update skip-gram model by training on a single sentence.

#     The sentence is a list of Vocab objects (or None, where the corresponding
#     word is not in the vocabulary. Called internally from `Word2Vec.train()`.

#     This is the non-optimized, Python version. If you have cython installed, gensim
#     will use the optimized version from word2vec_inner instead.

#     """
#     print sentence_id


#     if model.negative:
#         # precompute negative labels
#         labels = zeros(model.negative + 1)
#         labels[0] = 1.0

#     for pos, word in enumerate(sentence):
#         if word is None:
#             continue  # OOV word in the input sentence => skip
#         # reduced_window = random.randint(model.window)  # `b` in the original word2vec code
#         reduced_window = 0
#         # now go over all words from the (reduced) window, predicting each one in turn
#         start = max(0, pos - model.window + reduced_window)
#         for pos2, word2 in enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start):
#             # don't train on OOV words and on the `word` itself
#             if word2 and not (pos2 == pos):
#                 l1 = model.syn0[word2.index]

#                 # print "l1.shape", l1.shape
#                 word_vec_length = l1.shape[0]
#                 doc_vec = model.doc[sentence_id]
#                 doc_vec_length = model.doc_vec_size

                
#                 if model.cbow_type == 1:
#                     # average mode
#                     neu1e = zeros((doc_vec_length),dtype=REAL)
#                     l1 = (l1+doc_vec)/2
#                 else:
#                     neu1e = zeros((word_vec_length+doc_vec_length),dtype=REAL)
#                     l1 = numpy.append(l1,doc_vec)


                
#                 # print "l1.shape", l1.shape
#                 # neu1e = zeros(l1.shape)

#                 if model.hs:
#                     # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
#                     l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
#                     # print "l2a.shape",l2a.shape
#                     fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
#                     ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
#                     model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
#                     neu1e += dot(ga, l2a) # save error

#                 if model.negative:
#                     # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
#                     word_indices = [word.index]
#                     while len(word_indices) < model.negative + 1:
#                         w = model.table[numpy.random.randint(model.table.shape[0])]
#                         if w != word.index:
#                             word_indices.append(w)
#                     l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
#                     fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
#                     gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
#                     model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
#                     neu1e += dot(gb, l2b) # save error

#                 if model.cbow_type == 1:
#                     # average mode

#                     if not model.is_using_word2vec and model.freeze_learn:
#                         model.syn0[word2.index] += neu1e
#                     model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)
#                 else:    

#                     if not model.is_using_word2vec and model.freeze_learn:
#                         model.syn0[word2.index] += neu1e[:word_vec_length] # learn input -> hidden, here for all words in the window separately
#                     model.doc[sentence_id]    += neu1e[word_vec_length:]*(alpha_doc/alpha)
#                     # model.syn0[word2.index] += neu1e  # learn input -> hidden

#     return len([word for word in sentence if word is not None])


# # cbow_type == 3
'''隠れ層を並び順を考慮した学習をする'''
def train_sentence_cbow_concatenate(model, sentence_id, sentence, alpha, work=None, neu1=None,alpha_doc=0.025):
    """
    Update CBOW model by training on a single sentence.

    """
    # print sentence_id


    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window) # `b` in the original word2vec code
        # reduced_window = 0 #固定ウィンドウ幅で計算する

        start = max(0, pos - model.window)
        end   = pos+model.window_r+1
        window_pos = enumerate(sentence[start : end], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
        word2_indices_top = [-1 for i in xrange(model.window-pos)]
        word2_indices_bottom = [-1 for i in xrange(pos + model.window_r-len(sentence)+1)]

        
        # NULL文字を追加する必要性がある 
        #
        # for pos2, word2 in enumerate(sentence[start : end], start):
        # print [word2.index for pos2, word2 in enumerate(sentence)]
        # print word2_indices_top
        # print word2_indices
        # print word2_indices_bottom
        # print len(word2_indices_top)+len(word2_indices)+len(word2_indices_bottom)
        # print len(word2_indices_top)
        # print len(word2_indices)
        # print len(word2_indices_bottom)

        def get_wordvec_with_null(index):
            # return numpy.zeros(model.layer1_size) if index == -1 else model.syn0[index]
            return model.null_vec if index == -1 else model.syn0[index]


        # word_vec_length = model.layer1_size
        # word_vec_length_total = word_vec_length*model.window
        l1 = numpy.array([get_wordvec_with_null(index) for index in word2_indices_top+word2_indices+word2_indices_bottom])
        null_index = [i for i,index in enumerate(word2_indices_top+word2_indices+word2_indices_bottom) if index == -1]
        # l1 = model.syn0[word2_indices]
        # print l1
        # print "l1.shape",l1.shape
        # l1_shape = l1.shape # (5,200)
        # print "l1.shape",l1.shape
        # print "l1.dtype",l1.dtype

        # print "l1.dtype",l1.dtype
        # word_vec_length_total = word_vec_length*len(word2_indices)
        word_vec_length_total = l1.shape[0]*l1.shape[1]
        # print "word_vec_length_total",word_vec_length_total
        # input_word_num = l1.shape[0]
        l1 = numpy.reshape(l1,(word_vec_length_total,1))

        # if word2_indices and model.cbow_mean:
            # l1 /= len(word2_indices)

        # word_vec_length = l1.shape[0]
        # print "l1",l1.shape
        
        doc_vec = model.doc[sentence_id]
        doc_vec_length = model.doc_vec_size

        neu1e = zeros((word_vec_length_total+doc_vec_length),dtype=REAL)

        l1 = numpy.append(l1,doc_vec)

        # print "l1 slow : ", 
        # if sentence_id == 0:
        #     for iii in range(len(l1)):
        #         print l1[iii],
        # doc_index = len(model.vocab)+sentence_id
        if model.hs:
            # l2a = deepcopy(model.syn1[word.point][:,:word_vec_length_total])
            # l2a_doc = deepcopy(model.syn1[word.point][:,-doc_vec_length:])
            # l2a = numpy.append(l2a,l2a_doc,axis=1)
            l2a = deepcopy(model.syn1[word.point])

            


            
            # print "l1.shape",l1.shape
            # print "l2a.shape",l2a.shape
            fa = 1. / (1. + exp(-dot(l1, l2a.T))) # propagate hidden -> output
            # if sentence_id == 0:
            #     print "fa : ", fa

            ga = (1. - word.code - fa) * alpha # vector of error gradients multiplied by the learning rate
            # print outer(ga, l1).shape
            outer_result = outer(ga, l1)
            # print model.syn1[word.point].l1_shape
            model.syn1[word.point] += outer_result
            # print outer(ga, l1)[:,word_vec_length:]
            # print dot(ga, l2a)
            neu1e += dot(ga, l2a) # save error

            # if sentence_id == 0:
            #     # print "word.point : ", len(word.point)
            #     for iii in range(len(neu1e)):
            #         print neu1e[iii],
            #     # print "l2a"
            #     # print "word.point ", word.point
            #     # for iii in range(len(l2a)):
            #     #     print l2a[iii],
            #     print "" 



        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)

            word_indices = [word.index]
            while len(word_indices) < model.negative + 1:
                w = model.table[numpy.random.randint(model.table.shape[0])]
                if w != word.index:
                    word_indices.append(w)
            # l2b = deepcopy(model.syn1neg[word_indices][:,:word_vec_length_total])
            # l2b_doc = deepcopy(model.syn1neg[word_indices][:,-doc_vec_length:])
            # l2b = numpy.append(l2b,l2b_doc,axis=1)
            # l2b = model.syn1neg[word_indices]
            l2b = deepcopy(model.syn1neg[word_indices])
            # print l2b.shape
            fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output


            if sentence_id == 0:
                print "fb : ", fb

            gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
            outer_result = outer(gb, l1)
            model.syn1neg[word_indices] += outer_result
            neu1e += dot(gb, l2b) # save error
            

        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        # print "word2_indices",word2_indices
        # print "neu1e", neu1e.shape
        # print " model.syn0[word2_indices]" ,  model.syn0[word2_indices].shape
        # print neu1e[:word_vec_length_total].shape
        # print (len(word2_indices),word_vec_length)
        syn0_buf = neu1e[:word_vec_length_total]
        # print "syn0_buf",syn0_buf.shape
        # print syn0_buf

        resize_shape = model.syn0[word2_indices].shape
        start_index_syn0 = len(word2_indices_top)*model.layer1_size
        end_index_syn0   = len(syn0_buf)-len(word2_indices_bottom)*model.layer1_size

        syn0_buf_flat = numpy.reshape(syn0_buf[start_index_syn0:end_index_syn0],resize_shape)
        # for null_i in null_index:

        # syn0_buf_flat_null = numpy.reshape(syn0_buf[start_index_syn0:end_index_syn0],resize_shape)

        # print len(word2_indices)
        if not model.freeze_learn:
            model.syn0[word2_indices] += syn0_buf_flat # learn input -> hidden, here for all words in the window separately
            # model.null_vec            += syn0_buf_flat_null
        model.doc[sentence_id]    += neu1e[-doc_vec_length:]*(alpha_doc/alpha)
        # print neu1e[word_vec_length:]
    # print model.doc 
    # if sentence_id == 0:
        # print model.test_vec
        # print "*"*500
        # print model.doc[0]


    return len([word for word in sentence if word is not None])


# cbow_type == 2
'''隠れ層を並び順を考慮した学習をする + syn1_doc'''
def train_sentence_cbow_concatenate_syn1_doc(model, sentence_id, sentence, alpha, work=None, neu1=None,alpha_doc=0.025):
    """
    Update CBOW model by training on a single sentence.

    """
    # print sentence_id


    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window) # `b` in the original word2vec code
        # reduced_window = 0 #固定ウィンドウ幅で計算する
        # start = max(0, pos - model.window + reduced_window)
        # window_pos = enumerate(sentence[start : pos + model.window - reduced_window], start)
        # word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
        # window = 2*model.window
        # half_window = model.window


        start = max(0, pos - model.window)
        end   = pos+model.window_r+1
        # コメントアウト
        # if pos - half_window < 0:
        #     start = 0
        #     end   = window + 1
        # if pos - half_window >= 0:
        #     start = pos - half_window
        #     end   = pos + half_window + 1

        # if pos + half_window >= len(sentence):
        #     end = len(sentence)
        #     start = max(0, end - window - 1)


        word2_indices_top = [-1 for i in xrange(model.window-pos)]
        word2_indices_bottom = [-1 for i in xrange(pos + model.window_r-len(sentence)+1)]



        # start = max(0, pos - half_window)
        # end   = pos + model.window + 1 if start == 0 else pos + model.window
        # end_last = len(sentence)
        # start = max(0, pos - model.window - (end_last - pos)) if end_last - pos < half_window else start
        window_pos = enumerate(sentence[start : end], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

        # print "len",len(word2_indices)
        # continue
        # l1 = np_sum(model.syn0[word2_indices], axis=0) # 1 x layer1_size

        # word_vec_length = model.layer1_size
        # word_vec_length_total = word_vec_length*model.window


        def get_wordvec_with_null(index):
            return model.null_vec if index == -1 else model.syn0[index]

        # word_vec_length = model.layer1_size
        # word_vec_length_total = word_vec_length*model.window
        l1 = numpy.array([get_wordvec_with_null(index) for index in word2_indices_top+word2_indices+word2_indices_bottom])

        # l1 = model.syn0[word2_indices]


        # l1_shape = l1.shape # (5,200)
        # print "l1.shape",l1.shape
        # print "l1.dtype",l1.dtype

        # print "l1.dtype",l1.dtype
        # word_vec_length_total = word_vec_length*len(word2_indices)
        word_vec_length_total = l1.shape[0]*l1.shape[1]
        # print "word_vec_length_total",word_vec_length_total
        # input_word_num = l1.shape[0]
        l1 = numpy.reshape(l1,(word_vec_length_total,1))

        # if word2_indices and model.cbow_mean:
            # l1 /= len(word2_indices)

        # word_vec_length = l1.shape[0]
        # print "l1",l1.shape
        
        doc_vec = model.doc[sentence_id]
        doc_vec_length = model.doc_vec_size

        neu1e = zeros((word_vec_length_total+doc_vec_length),dtype=REAL)

        l1 = numpy.append(l1,doc_vec)
        # l1 = list(l1)+list(doc_vec)
        # print "l1.dtype",l1.dtype
        # l1 = array(l1,dtype=REAL)
        # print "l1.dtype",l1.dtype
        # hidden_size = word_vec_length+doc_vec_length
        
        # doc_index = len(model.vocab)+sentence_id
        if model.hs:

            l2a = deepcopy(model.syn1[word.point][:,:word_vec_length_total]) # 2d matrix, codelen x layer1_size
            # print "l2a.shape",l2a.shape
            # l2a.resize(l2a.shape[0]*word_vec_length_total)
            l2a_doc = deepcopy(model.syn1_doc[sentence_id])
            l2a_doc.resize((l2a.shape[0],doc_vec_length))

            # print "l2a_doc.shape",l2a_doc.shape
            l2a = numpy.append(l2a,l2a_doc,axis=1)

            # print "l1.shape",l1.shape
            # print "l2a.T.shape",l2a.T.shape
            # print model.syn1[len(model.vocab)+sentence_id].shape
            # print len(model.vocab)+sentence_id
            # l2a = list(l2a)+list(model.syn1[len(model.vocab)+sentence_id])
            # l2a = array(l2a)
            # print l2a.shape
            # print l1
            # print l2a
            # print l1.dtype
            # print l2a.T.dtype
            # print dot(l1, l2a.T).shape
            # print np_sum(-dot(l1, l2a.T))
            # print -dot(l1, l2a.T).shape
            # print exp(-dot(l1, l2a.T))

            fa = 1. / (1. + exp(-dot(l1, l2a.T))) # propagate hidden -> output
            ga = (1. - word.code - fa) * alpha # vector of error gradients multiplied by the learning rate
            # print outer(ga, l1).shape
            # print word_vec_length
            # print "ga.shape",ga.shape
            # print "l1.shape",l1.shape
            # print outer(ga, l1).shape
            # print "word_vec_length_total",word_vec_length_total
            outer_result = outer(ga, l1)
            # print sum(outer(ga, l1)[0,:])
            # print sum(outer(ga, l1)[1,:])
            # print sum(outer(ga, l1)[2,:])
            # print outer_result.shape
            # print "model.syn1[word.point].shape",model.syn1[word.point].shape
            # print "outer_result[:,:word_vec_length_total].shape",outer_result[:,:word_vec_length_total].shape
            # print outer_result[0,:word_vec_length_total].T.shape
            # print (outer_result.shape[0],word_vec_length)
            # print outer(ga, l1)[:,:word_vec_length].shape
            # print model.syn1[word.point].l1_shape
            # print type(outer_result[0,:word_vec_length_total])
            # syn1_buf = array(deepcopy(outer_result[:,:word_vec_length_total]),dtype=REAL)
            # print "syn1_buf.shape",syn1_buf.shape
            # resize_shape = (outer_result.shape[0],word_vec_length)
            model.syn1[word.point][:,:word_vec_length_total] += outer_result[:,:word_vec_length_total] # learn hidden -> output
            if model.is_np_mean_syn1:
                model.syn1_doc[sentence_id]  += np_mean(outer_result[:,word_vec_length_total:],axis=0)*(alpha_doc/alpha)
            else:
                model.syn1_doc[sentence_id]  += np_sum(outer_result[:,word_vec_length_total:],axis=0)*(alpha_doc/alpha)

            # model.syn1_doc[sentence_id]  += outer_result[0,word_vec_length_total:]*(alpha_doc/alpha)

            # print outer(ga, l1)[:,word_vec_length:]
            # print "dot(ga, l2a)",dot(ga, l2a).shape
            # print ga.shape
            # print fa.shape
            # print "neu1e",neu1e.shape
            # print "dot(ga, l2a)",dot(ga, l2a).shape
            # print dot(ga, l2a)
            neu1e += dot(ga, l2a) # save error

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [word.index]
            while len(word_indices) < model.negative + 1:
                w = model.table[numpy.random.randint(model.table.shape[0])]
                if w != word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices][:,:word_vec_length_total] # 2d matrix, k+1 x layer1_size
            # print model.syn1neg_doc.shape
            l2b_doc = deepcopy(model.syn1neg_doc[sentence_id])
            l2b_doc.resize((l2b.shape[0],doc_vec_length))
            l2b = numpy.append(l2b,l2b_doc,axis=1)
            # print l1.shape
            # print l2b.shape
            fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
            gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
            outer_result = outer(gb, l1)
            model.syn1neg[word_indices][:,:word_vec_length_total] += outer_result[:,:word_vec_length_total]# learn hidden -> output
            if model.is_np_mean_syn1:
                model.syn1neg_doc[sentence_id] += np_mean(outer_result[:,word_vec_length_total:],axis=0)*(alpha_doc/alpha)
            else:
                model.syn1neg_doc[sentence_id] += np_sum(outer_result[:,word_vec_length_total:],axis=0)*(alpha_doc/alpha)
            # model.syn1neg_doc[sentence_id] += outer_result[0,word_vec_length_total:]*(alpha_doc/alpha)
            # print "neu1e",neu1e.shape
            # print "dot(gb, l2b)",dot(gb, l2b).shape
            # print dot(gb, l2b)
            neu1e += dot(gb, l2b) # save error
            

        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        # print "word2_indices",word2_indices
        # print "neu1e", neu1e.shape
        # print " model.syn0[word2_indices]" ,  model.syn0[word2_indices].shape
        # print neu1e[:word_vec_length_total].shape
        # print (len(word2_indices),word_vec_length)
        syn0_buf = neu1e[:word_vec_length_total]
        # print syn0_buf.shape
        # print syn0_buf
        # print type(syn0_buf)
        # print syn0_buf.dtype


        # resize_shape = model.syn0[word2_indices].shape
        # print len(word2_indices)
        # print resize_shape
        # print syn0_buf.shape



        resize_shape = model.syn0[word2_indices].shape
        start_index_syn0 = len(word2_indices_top)*model.layer1_size
        end_index_syn0   = len(syn0_buf)-len(word2_indices_bottom)*model.layer1_size

        syn0_buf_flat = numpy.reshape(syn0_buf[start_index_syn0:end_index_syn0],resize_shape)
        # print len(word2_indices)
        if not model.freeze_learn:
            model.syn0[word2_indices] += syn0_buf_flat # learn input -> hidden, here for all words in the window separately
        model.doc[sentence_id]    += neu1e[-doc_vec_length:]*(alpha_doc/alpha)


        # if not model.is_using_word2vec:
        #     model.syn0[word2_indices] += numpy.reshape(syn0_buf,resize_shape) # learn input -> hidden, here for all words in the window separately
        # model.doc[sentence_id]    += neu1e[-doc_vec_length:]*(alpha_doc/alpha)
        # print neu1e[word_vec_length:]
    # print model.doc 
    # if sentence_id == 0:
        # print model.test_vec
        # print "*"*500
        # print model.doc[0]


    return len([word for word in sentence if word is not None])


'''隠れ層はベクトルの和or平均'''
# cbow_type == 1
def train_sentence_cbow_average_simple(model, sentence_id, sentence, alpha, work=None, neu1=None,alpha_doc=0.025):
    """
    Update CBOW model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    # print sentence_id


    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window) # `b` in the original word2vec code
        reduced_window = 0 #固定ウィンドウ幅で計算する
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
        l1 = np_sum(model.syn0[word2_indices], axis=0) # 1 x layer1_size

        if word2_indices and model.cbow_mean and model.average_flag == 0:
            l1 /= len(word2_indices)

        word_vec_length = model.layer1_size
        # print "l1",l1.shape
        
        doc_vec = model.doc[sentence_id]
        doc_vec_length = model.doc_vec_size

        neu1e = zeros((doc_vec_length))

        # l1 = numpy.append(l1,doc_vec)
        l1 = l1 + doc_vec

        if word2_indices and model.cbow_mean and model.average_flag == 1:
            l1 = l1/(len(word2_indices)+1)
        # l1 = list(l1)+list(doc_vec)
        # l1 = array(l1)
        # hidden_size = word_vec_length+doc_vec_length
        
        # doc_index = len(model.vocab)+sentence_id
        if model.hs:
            l2a = deepcopy(model.syn1[word.point]) # 2d matrix, codelen x layer1_size

            # l2a_doc = deepcopy(model.syn1_doc[sentence_id])
            # l2a_doc.resize((l2a.shape[0],doc_vec_length))
            # l2a = numpy.append(l2a,l2a_doc,axis=1)

            
            # print model.syn1[len(model.vocab)+sentence_id].shape
            # print len(model.vocab)+sentence_id
            # l2a = list(l2a)+list(model.syn1[len(model.vocab)+sentence_id])
            # l2a = array(l2a)
            # print l1.shape
            # print l2a.shape
            # print "l1.shape",l1.shape
            # print "l2a.shape",l2a.T.shape
            # print l1
            # print l2a
            # print sum(-dot(l1, l2a.T))

            fa = 1. / (1. + exp(-dot(l1, l2a.T))) # propagate hidden -> output
            # print "fa.shape ",fa.shape
            ga = (1. - word.code - fa) * alpha # vector of error gradients multiplied by the learning rate
            # print outer(ga, l1).shape
            # print word_vec_length
            # print outer(ga, l1).shape
            # print outer(ga, l1)[:,:word_vec_length].shape
            # print model.syn1[word.point].shape
            outer_result = outer(ga, l1)
            if not model.skip_id:
                model.syn1[word.point] += outer_result
            # model.syn1[word.point] += outer_result[:,:word_vec_length] # learn hidden -> output
            # model.syn1_doc[sentence_id] += outer_result[0,word_vec_length:]*(alpha_doc/alpha)

            # print outer(ga, l1).shape
            # print "dot(ga, l2a)",dot(ga, l2a).shape
            # print ga.shape
            # print fa.shape
            # print "neu1e",neu1e.shape
            neu1e += dot(ga, l2a) # save error

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [word.index]
            while len(word_indices) < model.negative + 1:
                w = model.table[numpy.random.randint(model.table.shape[0])]
                if w != word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
            # l2b_doc = deepcopy(model.syn1neg_doc[sentence_id])
            # l2b_doc.resize((l2b.shape[0],doc_vec_length))
            # l2b = numpy.append(l2b,l2b_doc,axis=1)
            # print l1.shape
            # print l2b.shape
            fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
            gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
            outer_result = outer(gb, l1)
            if not model.skip_id:
                model.syn1neg[word_indices] += outer_result
            # model.syn1neg[word_indices] += outer_result[:,:word_vec_length] # learn hidden -> output
            # model.syn1neg_doc[sentence_id]    += outer_result[0,word_vec_length:]*(alpha_doc/alpha)
            neu1e += dot(gb, l2b) # save error

        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        # print "word2_indices",word2_indices
        # print "neu1e", neu1e.shape
        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape

        # if not model.skip_id:

        if not model.freeze_learn:
            # model.syn0[word2_indices] += neu1e[:word_vec_length] # learn input -> hidden, here for all words in the window separately
            model.syn0[word2_indices] += neu1e
        # model.doc[sentence_id]    += neu1e[word_vec_length:]*(alpha_doc/alpha)
        model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)
        # print neu1e[word_vec_length:]
    # print model.doc 
    return len([word for word in sentence if word is not None])

# cbow_type == 4
'''隠れ層はベクトルの和or平均 syn1が単語ベクトルと文書ベクトルの長さである。単語と文書ベクトルを共に学習する'''
def train_sentence_cbow_average_plus_doc(model, sentence_id, sentence, alpha, work=None, neu1=None,alpha_doc=0.025):
    """
    Update CBOW model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    # print sentence_id



    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window) # `b` in the original word2vec code
        reduced_window = 0 #固定ウィンドウ幅で計算する
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
        l1 = np_sum(model.syn0[word2_indices], axis=0) # 1 x layer1_size


        if word2_indices and model.cbow_mean and model.average_flag == 0:
             l1 /= len(word2_indices)

        word_vec_length = model.layer1_size
        # print "l1",l1.shape
        
        doc_vec = model.doc[sentence_id]
        doc_vec_length = model.doc_vec_size

        neu1e = zeros((word_vec_length+doc_vec_length))

        l1 = numpy.append(l1,doc_vec)
        
        if word2_indices and model.cbow_mean and model.average_flag == 1:
            l1 = l1/(len(word2_indices)+1)

        # l1 = l1 + doc_vec
        # l1 = list(l1)+list(doc_vec)
        # l1 = array(l1)
        # hidden_size = word_vec_length+doc_vec_length
        
        # doc_index = len(model.vocab)+sentence_id
        if model.hs:
            l2a = deepcopy(model.syn1[word.point]) # 2d matrix, codelen x layer1_size



            fa = 1. / (1. + exp(-dot(l1, l2a.T))) # propagate hidden -> output
            # print "fa.shape ",fa.shape
            ga = (1. - word.code - fa) * alpha # vector of error gradients multiplied by the learning rate
            # print outer(ga, l1).shape
            # print word_vec_length
            # print outer(ga, l1).shape
            # print outer(ga, l1)[:,:word_vec_length].shape
            # print model.syn1[word.point].shape
            outer_result = outer(ga, l1)
            if not model.skip_id:
                model.syn1[word.point] += outer_result
            # model.syn1[word.point] += outer_result[:,:word_vec_length] # learn hidden -> output
            # model.syn1_doc[sentence_id] += outer_result[0,word_vec_length:]*(alpha_doc/alpha)

            # print outer(ga, l1).shape
            # print "dot(ga, l2a)",dot(ga, l2a).shape
            # print ga.shape
            # print fa.shape
            # print "neu1e",neu1e.shape
            neu1e += dot(ga, l2a) # save error

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [word.index]
            while len(word_indices) < model.negative + 1:
                w = model.table[numpy.random.randint(model.table.shape[0])]
                if w != word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
            # l2b_doc = deepcopy(model.syn1neg_doc[sentence_id])
            # l2b_doc.resize((l2b.shape[0],doc_vec_length))
            # l2b = numpy.append(l2b,l2b_doc,axis=1)
            # print l1.shape
            # print l2b.shape
            fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
            gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
            outer_result = outer(gb, l1)
            if not model.skip_id:
                model.syn1neg[word_indices] += outer_result
            # model.syn1neg[word_indices] += outer_result[:,:word_vec_length] # learn hidden -> output
            # model.syn1neg_doc[sentence_id]    += outer_result[0,word_vec_length:]*(alpha_doc/alpha)
            neu1e += dot(gb, l2b) # save error

        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        # print "word2_indices",word2_indices
        # print "neu1e", neu1e.shape
        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        # if not model.skip_id:

        if not model.freeze_learn:
            model.syn0[word2_indices] += neu1e[:word_vec_length] # learn input -> hidden, here for all words in the window separately
            # model.syn0[word2_indices] += neu1e
        model.doc[sentence_id]    += neu1e[word_vec_length:]*(alpha_doc/alpha)
        # model.doc[sentence_id]    += neu1e*(alpha_doc/alpha)
        # print neu1e[word_vec_length:]
    # print model.doc 
    return len([word for word in sentence if word is not None])


# cbow_type == 0
'''隠れ層はベクトルの和or平均 + syn1_doc  文書ベクトルの回帰パラメータを別で持ってくる'''
def train_sentence_cbow_syn1_doc(model, sentence_id, sentence, alpha, work=None, neu1=None,alpha_doc=0.025):
    """
    Update CBOW model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """

    # print sentence_id

    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # reduced_window = random.randint(model.window) # `b` in the original word2vec code
        reduced_window = 0 #固定ウィンドウ幅で計算する
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(sentence[start : pos + model.window_r + 1 - reduced_window], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
        l1 = np_sum(model.syn0[word2_indices], axis=0) # 1 x layer1_size

        if word2_indices and model.cbow_mean and model.average_flag == 0:
            l1 /= len(word2_indices)
        word_vec_length = model.layer1_size
        # print "l1",l1.shape
        
        doc_vec = model.doc[sentence_id]
        doc_vec_length = model.doc_vec_size

        neu1e = zeros((word_vec_length+doc_vec_length))

        l1 = numpy.append(l1,doc_vec)

        if word2_indices and model.cbow_mean and model.average_flag == 1:
            l1 = l1/ (len(word2_indices)+1)

        # l1 = list(l1)+list(doc_vec)
        # l1 = array(l1)
        # hidden_size = word_vec_length+doc_vec_length
        
        # doc_index = len(model.vocab)+sentence_id
        if model.hs:
            l2a = deepcopy(model.syn1[word.point]) # 2d matrix, codelen x layer1_size

            l2a_doc = deepcopy(model.syn1_doc[sentence_id])
            l2a_doc.resize((l2a.shape[0],doc_vec_length))
            l2a = numpy.append(l2a,l2a_doc,axis=1)

            
            # print model.syn1[len(model.vocab)+sentence_id].shape
            # print len(model.vocab)+sentence_id
            # l2a = list(l2a)+list(model.syn1[len(model.vocab)+sentence_id])
            # l2a = array(l2a)
            # print l2a.shape
            # print "l1.shape",l1.shape
            # print "l2a.shape",l2a.T.shape
            # print l1
            # print l2a
            # print sum(-dot(l1, l2a.T))
            fa = 1. / (1. + exp(-dot(l1, l2a.T))) # propagate hidden -> output
            # print "fa.shape ",fa.shape
            ga = (1. - word.code - fa) * alpha # vector of error gradients multiplied by the learning rate
            # print outer(ga, l1).shape
            # print word_vec_length
            # print outer(ga, l1).shape
            # print outer(ga, l1)[:,:word_vec_length].shape
            # print model.syn1[word.point].shape
            outer_result = outer(ga, l1)
            # model.syn1[word.point] += outer_result
            model.syn1[word.point] += outer_result[:,:word_vec_length] # learn hidden -> output
            if model.is_np_mean_syn1:
                model.syn1_doc[sentence_id] += np_mean(outer_result[:,word_vec_length:]*(alpha_doc/alpha))
            else:
                model.syn1_doc[sentence_id] += np_sum(outer_result[:,word_vec_length:]*(alpha_doc/alpha))
            # model.syn1_doc[sentence_id] += outer_result[0,word_vec_length:]*(alpha_doc/alpha)
            # model.syn1_doc[sentence_id] = np_mean
            # print "*"*20
            # print outer(ga, l1)[:,word_vec_length:].shape
            # print "-"*10
            # print outer(ga, l1)[:,:word_vec_length].shape

            # print "-"*10

            # print outer(ga, l1)[:,word_vec_length:]

            # print "-"*10
            # print outer(ga, l1)[:,:word_vec_length]
            # print "*"*20
            # print "dot(ga, l2a)",dot(ga, l2a).shape
            # print ga.shape
            # print fa.shape
            # print "neu1e",neu1e.shape
            neu1e += dot(ga, l2a) # save error

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [word.index]
            while len(word_indices) < model.negative + 1:
                w = model.table[numpy.random.randint(model.table.shape[0])]
                if w != word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
            l2b_doc = deepcopy(model.syn1neg_doc[sentence_id])
            l2b_doc.resize((l2b.shape[0],doc_vec_length))
            l2b = numpy.append(l2b,l2b_doc,axis=1)
            # print l1.shape
            # print l2b.shape
            fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
            gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
            outer_result = outer(gb, l1)
            # model.syn1neg[word_indices] += outer_result

            # print outer_result.shape
            model.syn1neg[word_indices] += outer_result[:,:word_vec_length] # learn hidden -> output
            if model.is_np_mean_syn1:
                model.syn1neg_doc[sentence_id] += np_mean(outer_result[:,word_vec_length:]*(alpha_doc/alpha))
            else:
                model.syn1neg_doc[sentence_id] += np_sum(outer_result[:,word_vec_length:]*(alpha_doc/alpha))

            # model.syn1neg_doc[sentence_id] += outer_result[0,word_vec_length:]*(alpha_doc/alpha)
            neu1e += dot(gb, l2b) # save error

        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        # print "word2_indices",word2_indices
        # print "neu1e", neu1e.shape
        # print "model.syn0[word2_indices]",model.syn0[word2_indices].shape
        if not model.freeze_learn:
            model.syn0[word2_indices] += neu1e[:word_vec_length] # learn input -> hidden, here for all words in the window separately
        model.doc[sentence_id]    += neu1e[word_vec_length:]*(alpha_doc/alpha)
        # print neu1e[word_vec_length:]
    # print model.doc 
    return len([word for word in sentence if word is not None])

