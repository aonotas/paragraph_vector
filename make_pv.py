#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import paragraph_vector
import pickle
from prettyprint import pp, pp_str

# Paragraph Vectorを作成する
def make_paragraph_vector(    
        vec_length=200,
        doc_vec_length = 200,
        min_count=20,
        window=12,
        window_r=12,
        negative=20,
        sg=0,          # 0:cbow , 1:skipgram
        cbow_type=1,  # 0:average_concat +syn1_doc 1:average , 2:concatenate+syn1_doc , 3:concatenate , 4:average_concat
        skip_gram_type=0, # 0:simple 1:average, 2:concatenate
        alpha=0.025,
        alpha_doc=0.025,
        alpha_rate = 0.99,
        alpha_flag=0,   # 0:学習パラメータは減少させていく, 1:減少させない
        cbow_mean=1, # 0:no 1:average
        iteration = 10, # 学習回数
        average_flag=0, # 0:word_vecの和のみ 1:doc_vecとword_vecの和
        is_np_mean_syn1=0, #0: syn1の誤差を平均， 1:syn1の誤差を合計 (cbow_type=0,2のとき)
        is_using_word2vec=0, # 1:word2vecの単語ベクトルを用いる, 0:用いない
        is_using_wiki = 0, # 0:none 1:wiki
        hs = 1, # 0:hsを使わない 1:使う
        sample=1e-5, # 1e-5 頻度の高い単語を減らす
        freeze_learn=0, # 0:更新する , 1: word2vecを使った場合に単語ベクトルを更新しない
        random_learn_flag=0, # 1:ランダムに学習する, 0:与えられた文章から順に学習する
        n_gram_mode = 0, # 0:変換しない 1,2,3-gram
        null_vec_type = 0, # 0:zeros , 1: ones, 2:random
        skip_concat = 0, # 0:not skip   1:nullをskip
        sentences=None,
        input_file=None
    ):

    import unicodedata
    # import uuid
    import datetime
    import zenhan

    # uniq_id = uuid.uuid4()
    d = datetime.datetime.today()
    uniq_id = d.strftime("%Y-%m-%d_%H:%M:%S")
    # 文字を正規化する
    def clean_text(text):
        # del_n = re.compile('\n')
        # text = del_n.sub('',text)
        text = text.lower()
        text = unicodedata.normalize('NFKC', text)
        text = zenhan.z2h(text,zenhan.ASCII|zenhan.DIGIT)
        return text

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

    print '...model setting'
    model = paragraph_vector.ParagraphVector(size=vec_length, doc_vec_size=doc_vec_length, min_count=min_count, window=window,window_r=window_r, sg=sg, negative=negative, cbow_type=cbow_type,cbow_mean=cbow_mean,average_flag=average_flag,alpha_flag=alpha_flag,is_np_mean_syn1=is_np_mean_syn1,is_using_word2vec=is_using_word2vec,hs=hs,skip_gram_type=skip_gram_type,sample=sample,freeze_learn=freeze_learn,random_learn_flag=random_learn_flag,null_vec_type=null_vec_type,skip_concat=skip_concat)

    print '..load input file & build vocab'
    sentences_length = model.build_vocab(sentences=open(input_file))
    if sentences_length > 0:
        sentences_length += 1
    alpha_doc_m = alpha_doc
    alpha_m = alpha
    for i in xrange(iteration):
        print "**iteration : %d " % i
        print "alpha %f" % alpha_doc_m
        model.train(input_file=input_file, alpha=alpha_m, alpha_doc=alpha_doc_m,sentences_length=sentences_length)


        save_filename = "trained_model_{}_iter{}.p".format(uniq_id,str(i))
        # model.save(save_filename)

        with open(save_filename, 'w') as output:
            pickle.dump(model, output)


        alpha_doc_m *= alpha_rate
        alpha_m       *= alpha_rate


    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vec_length', type=int, dest='vec_length', required=False, default=300)    
    parser.add_argument('--doc_vec_length', type=int, dest='doc_vec_length', required=False, default=300) 
    parser.add_argument('--min_count', type=int, dest='min_count', required=False, default=20) 
    parser.add_argument('--window', type=int, dest='window', required=False, default=10) 
    parser.add_argument('--window_r', type=int, dest='window_r', required=False, default=10) 
    parser.add_argument('--negative', type=int, dest='negative', required=False, default=20) 
    parser.add_argument('--sg', type=int, dest='sg', required=False, default=0)   # 0:cbow , 1:skipgram
    parser.add_argument('--cbow_type', type=int, dest='cbow_type', required=False, default=1)  # 0:average_concat +syn1_doc 1:average , 2:concatenate+syn1_doc , 3:concatenate , 4:average_concat
    parser.add_argument('--skip_gram_type', type=int, dest='skip_gram_type', required=False, default=0) 
    parser.add_argument('--alpha', type=float, dest='alpha', required=False, default=0.025) 
    parser.add_argument('--alpha_doc', type=int, dest='alpha_doc', required=False, default=0.025) 
    parser.add_argument('--alpha_rate', type=int, dest='alpha_rate', required=False, default=0.025) 
    parser.add_argument('--alpha_flag', type=int, dest='alpha_flag', required=False, default=0)  # 0:学習パラメータは減少させていく, 1:減少させない
    parser.add_argument('--cbow_mean', type=int, dest='cbow_mean', required=False, default=1)  # 0:no 1:average
    parser.add_argument('--iteration', type=int, dest='iteration', required=False, default=20) # 学習回数
    parser.add_argument('--average_flag', type=int, dest='average_flag', required=False, default=1) # 0:word_vecの和のみ 1:doc_vecとword_vecの和
    parser.add_argument('--is_np_mean_syn1', type=int, dest='is_np_mean_syn1', required=False, default=0) #0: syn1の誤差を平均， 1:syn1の誤差を合計 (cbow_type=0,2のとき)
    parser.add_argument('--is_using_word2vec', type=int, dest='is_using_word2vec', required=False, default=0) # 1:word2vecの単語ベクトルを用いる, 0:用いない
    parser.add_argument('--is_using_wiki', type=int, dest='is_using_wiki', required=False, default=0) # 0:none 1:wiki
    parser.add_argument('--hs', type=int, dest='hs', required=False, default=0) # 0:hsを使わない 1:使う
    parser.add_argument('--sample', type=float, dest='sample', required=False, default=1e-5)  # 1e-5 頻度の高い単語を減らす
    parser.add_argument('--freeze_learn', type=int, dest='freeze_learn', required=False, default=0) # 0:更新する  1: word2vecを使った場合に単語ベクトルを更新しない
    parser.add_argument('--random_learn_flag', type=int, dest='random_learn_flag', required=False, default=0) # 1:ランダムに学習する, 0:与えられた文章から順に学習する
    parser.add_argument('--n_gram_mode', type=int, dest='n_gram_mode', required=False, default=0) # 0:変換しない 1,2,3-gram
    parser.add_argument('--null_vec_type', type=int, dest='null_vec_type', required=False, default=2) # 0:zeros , 1: ones, 2:random
    parser.add_argument('--skip_concat', type=int, dest='skip_concat', required=False, default=0) # 0:not skip   1:nullをskip 
    parser.add_argument('--input', type=str, dest='input_file', required=False, default="INPUT.txt") 
    # 1行1ドキュメントのファイル名


    args = parser.parse_args()
    args_dict =  vars(args)
    pp(args_dict)



    make_paragraph_vector(**args_dict)