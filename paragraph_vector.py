#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Paragraph Vector

'''

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

from numpy import exp, dot, zeros, outer, dtype,get_include, float32 as REAL,uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, mean as np_mean, sum as np_sum
import numpy
logger = logging.getLogger("gensim.models.word2vec")

import random

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange

import unicodedata
import zenhan

# import pyximport; pyximport.install(pyimport = True)
# from inner_paragraph_vector import train_sentence_sg, train_sentence_cbow, FAST_VERSION
# from word2vec_inner import train_sentence_sg, train_sentence_cbow, FAST_VERSION

# numpy.seterr(all='ignore')


#random_seed
random_seed = 1234


# 文字を正規化する
def clean_text(text):
    # del_n = re.compile('\n')
    # text = del_n.sub('',text)
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = zenhan.z2h(text,zenhan.ASCII|zenhan.DIGIT)
    return text




fast_mode = 0 #0:no 1:fast

if fast_mode == 1:
    from inner_paragraph_vector import train_sentence_sg_simple, train_sentence_cbow_concatenate,train_sentence_cbow_average_simple,train_sentence_cbow_average_plus_doc,train_sentence_sg_average,train_sentence_sg_concat,FAST_VERSION
else:
    FAST_VERSION = -1
    print "FAST_VERSION : ", FAST_VERSION
    from slow_inner_paragraph_vector import train_sentence_sg_simple, train_sentence_sg_average,train_sentence_sg_concat,train_sentence_cbow_concatenate,train_sentence_cbow_concatenate_syn1_doc,train_sentence_cbow_average_simple,train_sentence_cbow_average_plus_doc,train_sentence_cbow_syn1_doc


class ParagraphVector(object):
    """docstring for ParagraphVector"""
    def __init__(self, sentences=None, size=100, doc_vec_size=None, window=5,window_r=5, min_count=5,sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, cbow_type = 2, hs=1, negative=0, cbow_mean=0 , skip_id=0,average_flag=0,alpha_flag=0,is_np_mean_syn1=1,is_using_word2vec=0,skip_gram_type=0 ,freeze_learn=1,random_learn_flag=1,null_vec_type=0,skip_concat=0,is_using_label=0):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=1`), skip-gram is used. Otherwise, `cbow` is employed.
        `word_vec_size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
        `seed` = for the random number generator.
        `min_count` = ignore all words with total frequency lower than this.
        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.
        `workers` = use this many worker threads to train the model (=faster training with multicore machines)
        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0)
        `negative` = if > 0, negative sampling will be used, the int for negative
                specifies how many "noise words" should be drawn (usually between 5-20)
        `cbow_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
                Only applies when cbow is used.
        `skip_id` = 途中からモデルを学習する際に利用する。
        """
        if doc_vec_size is None:
            doc_vec_size = size
        if sentences is None:
            self.doc_len = 0
        else:
            self.doc_len = len(sentences)# 文章数をカウント
        print "self.cbow_type : ",cbow_type
        self.cbow_type = cbow_type
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.table = None # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.layer1_size = int(size)
        self.doc_vec_size = int(doc_vec_size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.window = int(window)
        self.window_r = int(window_r)
        self.seed = seed
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.skip_id = skip_id
        self.average_flag = average_flag
        self.alpha_flag = alpha_flag
        self.is_np_mean_syn1 = is_np_mean_syn1
        self.is_using_word2vec = is_using_word2vec
        self.skip_gram_type = skip_gram_type
        self.random_learn_flag = random_learn_flag
        self.freeze_learn = freeze_learn
        self.null_vec_type = null_vec_type
        self.skip_concat = skip_concat
        self.is_using_label = is_using_label
        if sentences is not None:
            self.build_vocab(sentences)
            # self.train(sentences)

    def train_extra_sentences(self,extra_sentences=[],skip_id=180000):
        extra_len = len(extra_sentences)
        print "skip_id"+ str(skip_id)
        self.skip_id = skip_id
        # 文書ベクトルの行列を再確保する
        self.doc_len = self.doc_len + extra_len
        bf_doc = deepcopy(self.doc)
        doc_new  = empty((self.doc_len, self.doc_vec_size), dtype=REAL)

        # 文書ベクトル学習済のものをコピー（numpyのextendの方が早い？）
        for i in xrange(skip_id):
            doc_new[i] = bf_doc[i]


        # 追加で学習する分を初期化
        for i in xrange(extra_len):
            index = self.skip_id + i
            doc_new[index] = (numpy.random.rand(self.doc_vec_size) - 0.5) / self.layer1_size

        self.doc = doc_new







    def make_table(self, table_size=100000000, power=0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.

        Called internally from `build_vocab()`.

        """
        logger.info("constructing a table with noise distribution from %i words" % len(self.vocab))
        # table (= list of words) of noise distribution for negative sampling
        vocab_size = len(self.index2word)
        self.table = zeros(table_size, dtype=uint32)

        if not vocab_size:
            logger.warning("empty vocabulary in word2vec, is this intended?")
            return

        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.vocab[self.index2word[widx]].count**power / train_words_pow
        for tidx in xrange(table_size):
            self.table[tidx] = widx
            if 1.0 * tidx / table_size > d1:
                widx += 1
                d1 += self.vocab[self.index2word[widx]].count**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1

    
    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words" % len(self.vocab))

        # build the huffman tree
        heap = self.vocab.values()
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()# points : 最初は空のリスト
                # print "index : ",node.index
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)# root からのノードのindex（グループ化？   ）
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i" % max_depth)

    def precalc_sampling(self):
        """Precalculate each vocabulary item's threshold for sampling"""
        if self.sample:
            logger.info("frequent-word downsampling, threshold %g; progress tallies will be approximate" % (self.sample))
            total_words = sum(v.count for v in itervalues(self.vocab))
            threshold_count = float(self.sample) * total_words
        for v in itervalues(self.vocab):
            prob = (sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if self.sample else 1.0
            v.sample_probability = min(prob, 1.0)


    def build_vocab(self, sentences, reset_flag=True):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of utf8 strings.

        """
        logger.info("collecting all words and their counts")
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            sentence = sentence.split(u" ")
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                    (sentence_no, total_words, len(vocab)))
            for word in sentence:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i word types from a corpus of %i words and %i sentences" %
            (len(vocab), total_words, sentence_no + 1))

        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, v in iteritems(vocab):
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.index2word.append(word)
                self.vocab[word] = v
        logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_table()
        # precalculate downsampling thresholds
        self.precalc_sampling()
        if reset_flag:
            self.reset_weights()

    def set_syn0(self, syn0):
        self.syn0 = syn0

    def set_syn1(self, syn1):
        if hasattr(self,"syn1"):
            self.syn1[:,:self.syn1_word_size] = syn1[:,:]
        if hasattr(self,"syn1neg"):
            self.syn1neg[:,:self.syn1_word_size] = syn1[:,:]

    def train(self, sentences, total_words=None, word_count=0, chunksize=100 , alpha=0.025, alpha_doc=0.025):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of utf8 strings.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("Cython compilation failed, training will be slow. Do you have Cython installed? `pip install cython`")
        logger.info("training model with %i workers on %i vocabulary and %i features, "
            "using 'skipgram'=%s 'hierarchical softmax'=%s 'subsample'=%s and 'negative sampling'=%s" %
            (self.workers, len(self.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        
        '''学習率を設定'''
        self.alpha_doc = float(alpha_doc)
        self.alpha = float(alpha)

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.syn1_size, dtype=REAL)  # each thread must have its own work memory
            # neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            neu1 = zeros(self.syn1_size, dtype=REAL) 

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break

                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                if self.alpha_flag == 1:
                    alpha = self.alpha
                # print "alpha", alpha
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                if self.sg:

                    if self.skip_gram_type == 0:
                        # job_id = 0
                        # sentence_id_,_ = job[job_id]
                        # print "py sentence_id = ",sentence_id_
                        # bf = deepcopy(self.doc[sentence_id_][0])
                        # print "bf : ",bf
                        job_words = sum(train_sentence_sg_simple(self, sentence_id,sentence, alpha, work,self.alpha_doc) for sentence_id,sentence in job)
                        # print "af : ",self.doc[sentence_id_][0]
                        # print "re : ", self.doc[sentence_id_][0] - bf
                    elif self.skip_gram_type == 1:

                        # ids_back = [sentence_id for sentence_id,_ in job]
                        # bf_ = deepcopy(self.doc[ids_back])


                        job_words = sum(train_sentence_sg_average(self, sentence_id,sentence, alpha, work, neu1 ,self.alpha_doc) for sentence_id,sentence in job)

                        # af_ = self.doc[ids_back]
                        # print numpy.mean(af_ - bf_ )

                    elif self.skip_gram_type == 2:

                        ids_back = [sentence_id for sentence_id,_ in job]
                        bf_ = deepcopy(self.doc[ids_back])

                        job_words = sum(train_sentence_sg_concat(self, sentence_id,sentence, alpha, work,neu1,self.alpha_doc) for sentence_id,sentence in job)

                        af_ = self.doc[ids_back]
                        print numpy.mean(af_ - bf_ )

                elif self.cbow_type == 4:

                    ids_back = [sentence_id for sentence_id,_ in job]
                    bf_ = deepcopy(self.doc[ids_back])

                    # job_words = sum(train_sentence_cbow_average_plus_doc_vec_extra_train(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)
                    job_words = sum(train_sentence_cbow_average_plus_doc(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)


                    af_ = self.doc[ids_back]
                    print numpy.mean(af_ - bf_ )

                    # print "re : ", af_ - bf_
                # elif self.cbow_type == 5:
                #     job_words = sum(train_sentence_cbow_concatenate_v2(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)
                elif self.cbow_type == 3:
                    job_id = 0
                    ids_back = [sentence_id for sentence_id,_ in job]
                    bf_ = deepcopy(self.doc[ids_back])
                    sentence_id_,sentence_ = job[job_id]
                    # # print "py sentence_id = ",sentence_id_
                    # bf = deepcopy(self.doc[sentence_id_])
                    # print "bf : ",bf
                    # print "null_vec", self.null_vec
                    job_words = sum(train_sentence_cbow_concatenate(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)

                    # af_ = self.doc[ids_back]
                    # print numpy.mean(af_ - bf_ )
                    # print "af : ",self.doc[sentence_id_]
                    # print "re : ", self.doc[sentence_id_] - bf


                    # print sum(self.doc[sentence_id_] - bf)
                elif self.cbow_type == 2:
                    job_words = sum(train_sentence_cbow_concatenate_syn1_doc(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)
                elif self.cbow_type == 1:

                    ids_back = [sentence_id for sentence_id,_ in job]
                    bf_ = deepcopy(self.doc[ids_back])
                    job_words = sum(train_sentence_cbow_average_simple(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)


                    af_ = self.doc[ids_back]
                    print numpy.mean(af_ - bf_ )
                    # print af_ - bf_ 


                elif self.cbow_type == 0:
                    job_words = sum(train_sentence_cbow_syn1_doc(self, sentence_id,sentence, alpha, work, neu1,self.alpha_doc) for sentence_id,sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                            (100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():

            '''
                ここでsentencesのindexをradom.shuffleしランダムに学習していく
            '''

            indexes_sentence_ids = numpy.array(range(len(sentences)))

            if self.random_learn_flag:
                random.shuffle(indexes_sentence_ids, lambda: random_seed)
                #numpy.random.shuffle(indexes_sentence_ids)

            for index in xrange(len(sentences)):
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sentence_id = indexes_sentence_ids[index]
                sentence    = sentences[sentence_id]
                sentence = sentence.split(u" ")

                # 途中まで学習している場合はスキップする（学習済モデルから追加で学習する場合）
                if sentence_id < self.skip_id:
                    print "skip! :"+str(sentence_id) +" "+str(self.skip_id)
                    continue

                sampled = [self.vocab[word] for word in sentence
                    if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= numpy.random.random_sample())]
                yield (sentence_id,sampled)

        # no_oov = ([self.vocab.get(word, None) for word in sentence] for sentence in sentences)
        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]


    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        numpy.random.seed(self.seed)
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        self.doc  = empty((self.doc_len, self.doc_vec_size), dtype=REAL)

        

        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            self.syn0[i] = (numpy.random.rand(self.layer1_size) - 0.5) / self.layer1_size
        for i in xrange(self.doc_len):
            self.doc[i] = (numpy.random.rand(self.doc_vec_size) - 0.5) / self.layer1_size

        # syn1_size = self.layer1_size
        label_size = 0
        if self.is_using_label:
            label_size = 10

        if self.cbow_type == 0:
            syn1_size = self.layer1_size+label_size
            syn1_word_size = self.layer1_size
        if self.cbow_type == 1:
            # syn1_size = self.layer1_size+self.doc_vec_size
            syn1_size = self.layer1_size+label_size
            syn1_word_size = self.layer1_size
        if self.cbow_type == 2:
            syn1_size = self.layer1_size*self.window+self.layer1_size*self.window_r+label_size
            syn1_word_size = self.layer1_size*self.window+self.layer1_size*self.window_r
        if self.cbow_type == 3:
            syn1_size = self.layer1_size*self.window+self.layer1_size*self.window_r+self.doc_vec_size+label_size
            syn1_word_size = self.layer1_size*self.window+self.layer1_size*self.window_r
        if self.cbow_type == 4:
            syn1_size = self.layer1_size+self.doc_vec_size+label_size
            syn1_word_size = self.layer1_size
        # if self.cbow_type == 5:
        #     syn1_size = self.layer1_size
        if self.sg == 1:
            if self.skip_gram_type == 0:
                # simple
                syn1_size = self.layer1_size+label_size
            elif self.skip_gram_type == 1:
                # average
                syn1_size = self.layer1_size+label_size
            elif self.skip_gram_type == 2:
                # concat
                syn1_size = self.layer1_size+self.doc_vec_size+label_size


        self.adagrad_word = zeros((len(self.vocab),self.layer1_size ), dtype=REAL) # AdaGrad : Word
        self.adagrad_word_syn1 = zeros((len(self.vocab),syn1_size ), dtype=REAL) # AdaGrad : Word syn1
        self.adagrad_doc = zeros((len(self.doc),self.doc_vec_size ), dtype=REAL) # AdaGrad : Doc

        if self.hs:
            self.syn1 = zeros((len(self.vocab), syn1_size), dtype=REAL)
            if self.cbow_type == 0 or self.cbow_type == 2 or self.cbow_type == 5:
                self.syn1_doc = zeros((self.doc_len, self.doc_vec_size), dtype=REAL)
            # self.syn1_doc = zeros((self.doc_len, self.doc_vec_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), syn1_size), dtype=REAL)
            if self.cbow_type == 0 or self.cbow_type == 2 or self.cbow_type == 5:
                self.syn1neg_doc = zeros((self.doc_len, self.doc_vec_size), dtype=REAL)
            # self.syn1neg_doc = zeros((self.doc_len, self.doc_vec_size), dtype=REAL)

        # null_vec setting
        if self.null_vec_type == 0:
            null_vec = numpy.zeros((1, self.layer1_size), dtype=REAL)
        elif self.null_vec_type == 1:
            null_vec = numpy.ones((1, self.layer1_size), dtype=REAL)
        elif self.null_vec_type == 2:
            null_vec = (numpy.random.rand(self.layer1_size) - 0.5) / self.layer1_size


        if fast_mode == 1:
            self.null_vec = empty((1, self.layer1_size), dtype=REAL)
            self.null_vec[0] = null_vec
        else:
            self.null_vec = null_vec
            # print null_vec

        print self.null_vec
        self.syn1_word_size = syn1_word_size
        self.syn1_size = syn1_size
        # print "self.syn1_size : ",self.syn1_size
        self.syn0norm = None

        # self.test_vec = deepcopy(self.doc[0])

    def __getitem__(self, word):
        """
        Return a word's representations in vector space, as a 1D numpy array.

        Example::
          >>> trained_model['woman']
          array([ -1.40128313e-02, ...]

        """
        return self.syn0[self.vocab[word].index]


    def __contains__(self, word):
        return word in self.vocab


    def __str__(self):
        return "Word2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)

    '''パラメーターを保存する'''
    def save(self,file_path):
        savedata = [self.doc,self.syn0]
        if self.hs:
            savedata.append(self.syn1)
            if (self.cbow_type == 0 or self.cbow_type) == 2 and self.sg == 0:
                savedata.append(self.syn1_doc)

        if self.negative > 0:
            savedata.append(self.syn1neg)
            if (self.cbow_type == 0 or self.cbow_type) == 2 and self.sg == 0:
                savedata.append(self.syn1neg_doc)

        if self.hs == 1:
            print "len : ",len(savedata)
            with open(file_path, 'w') as output:
                pickle.dump([self.doc], output)
                # pickle.dump([self.doc, self.syn1[:-self.doc_vec_size]], output)
        else:
            with open(file_path, 'w') as output:
                pickle.dump([self.doc], output)


        # with open(file_path.replace(".pickle","")+"_long.pickle", 'w') as output:
        #     pickle.dump(savedata, output)

    def save_vocab(self,file_path):
        with open(file_path, 'w') as output:
            pickle.dump([self.vocab], output)


    '''パラメータを変更する'''
    def set_params(self,syn0=None,syn1=None,doc=None,syn1neg=None,syn1neg_doc=None,syn1_doc=None):
        if syn0 is not None:
            self.syn0 = syn0
        if syn1 is not None:
            self.syn1 = syn1
        if syn1neg is not None:
            self.syn1neg = syn1neg
        if syn1_doc is not None:
            self.syn1_doc = syn1_doc
        if syn1neg_doc is not None:
            self.syn1neg_doc = syn1neg_doc
        if doc is not None:
            self.doc = doc

    '''類似度計算'''
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def init_sims_doc(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'doc_norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.doc.shape[0]):
                    self.doc[i, :] /= sqrt((self.doc[i, :] ** 2).sum(-1))
                self.doc_norm = self.doc
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.doc_norm = (self.doc / sqrt((self.doc ** 2).sum(-1))[..., newaxis]).astype(REAL)


    def most_similar(self, positive=[], negative=[], topn=10):

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]
        # print "simi"
        positive = [clean_text(w) for w in positive]
        negative = [clean_text(w) for w in negative]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]


    def most_similar_doc(self, positive=[], negative=[], topn=10):

        self.init_sims_doc()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]
        print "simi"
        # positive = [clean_text(w) for w in positive]
        # negative = [clean_text(w) for w in negative]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(doc_id, 1.0)  for doc_id in positive]
        negative = [(doc_id, -1.0) for doc_id in negative]

        # compute the weighted average of all words
        all_docs, mean = set(), []
        for doc_id, weight in positive + negative:
            mean.append(weight * self.doc_norm[doc_id])
            all_docs.add(doc_id)

        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.doc_norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_docs)]
        # ignore (don't return) words from the input
        result = [(sim, float(dists[sim])) for sim in best if sim not in all_docs]
        return result[:topn]

class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"

def load_text_file(filename="corpus/wakati_pn_uniq_th.txt",show=0):
    f = open(filename)
    line = f.readline()
    sentences = []
    while line:
        line = line.decode("utf-8")
        line = clean_text(line)
        sentence = [i for i in line.split(" ")]
        sentences.append(sentence)
        if show:
            print len(sentences),
            print len(sentence)
        # sentences.append()
        # print line
        # print "*"*35
        line = f.readline()
    return sentences

# データをクロスバリデーションを変換する
def data_cross_validation(data=[],N=10,n=0):
    trainData = [d for i, d in enumerate(data) if i % N != n]
    testData  = [d for i, d in enumerate(data) if i % N == n]
    return [trainData,testData]

# データをシャッフルする
def data_shuffle(data=[]):
    data2 = data[:]
    seed = 0.444298694171234
    random.shuffle(data2, lambda: seed)
    return data2


def test_vec_quality(filename):
    # vecファイルを読み込む
    def load_vec_file(file_path):
        with open(file_path, 'r') as f:
                vec_data = pickle.load(f)
        return vec_data

    import paragraph_vector
    import pickle

    model = paragraph_vector.ParagraphVector()
    filename = "th_doc_vec_cbow_alpha_0.050000_loop1_200_400_1_12_20_0_1_1_.pickle"
    doc_vecs = load_vec_file(filename)[0]

    model.doc = doc_vecs

    doc_filename = "corpus/wakati_pn_uniq_th.txt"
    doc_text = [f.strip() for f in open(doc_filename, "rU")]

    def new_ids():
        positive_end=100000
        N = 10
        n = 0
        new_sentences = []
        ids = range(len(doc_text))
        p_sentences = ids[:positive_end]
        n_sentences = ids[positive_end:]
        p_trainData,p_testData = paragraph_vector.data_cross_validation(p_sentences,N=N,n=n)
        n_trainData,n_testData = paragraph_vector.data_cross_validation(n_sentences,N=N,n=n)
        trainData = []
        trainData.extend(p_trainData)
        trainData.extend(n_trainData)
        testData = []
        testData.extend(p_testData)
        testData.extend(n_testData)
        new_sentences.extend(paragraph_vector.data_shuffle(trainData))
        new_sentences.extend(paragraph_vector.data_shuffle(testData))
        return new_sentences

    new_ids = new_ids()

    def simd(i):
        print doc_text[new_ids[i]]
        result = model.most_similar_doc([i])
        for index,x in enumerate(result):
            print index,
            print doc_text[new_ids[int(x[0])]] + "\t" + str(x[1])

    simd(1032)

if __name__ == '__main__':
    # Paragraph Vectorを作成する
    print ""