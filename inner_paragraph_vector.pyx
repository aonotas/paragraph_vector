#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t

DEF MAX_SENTENCE_LEN = 10000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil



''' sg simple'''
ctypedef void (*fast_sentence_sg_hs_ptr) (
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil
ctypedef unsigned long long (*fast_sentence_sg_neg_simple_ptr) (
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *doc, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil


''' cbow average'''
ctypedef void (*fast_sentence_cbow_average_simple_hs_ptr) (
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, REAL_t *doc, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, int average_flag) nogil
ctypedef unsigned long long (*fast_sentence_cbow_average_simple_neg_ptr) (
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, REAL_t *doc, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, int average_flag) nogil


''' cbow average plus doc'''
ctypedef void (*fast_sentence_cbow_average_plus_doc_hs_ptr) (
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, REAL_t *doc, const int size, const int doc_size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil
ctypedef unsigned long long (*fast_sentence_cbow_average_plus_doc_neg_ptr) (
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, REAL_t *doc, const int size, const int doc_size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil

''' cbow concat'''
ctypedef void (*fast_sentence_cbow_hs_concat_ptr) (
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *null_vec,const int window, const int window_r,const int sentence_len, REAL_t *l1, REAL_t *doc, REAL_t *syn0, REAL_t *syn1, 
    const int size,const int doc_size, const int l1_size, const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, int flag_freeze_learning) nogil #change
ctypedef unsigned long long (*fast_sentence_cbow_neg_concat_ptr) (
    const long sentence_id, const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *null_vec,const int window, const int window_r,const int sentence_len, REAL_t *l1, REAL_t *doc, REAL_t *syn0,
    REAL_t *syn1neg, const int size,const int doc_size, const int l1_size, 
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random,int flag_freeze_learning) nogil #change



''' cbow'''
ctypedef void (*fast_sentence_cbow_hs_ptr) (
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil
ctypedef unsigned long long (*fast_sentence_cbow_neg_ptr) (
    const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil


''' sg'''
ctypedef void (*fast_sentence_sg_hs_simple_ptr) (
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *doc, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil #modify!

ctypedef unsigned long long (*fast_sentence_sg_neg_ptr) (
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil




cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

''' cbow average'''
cdef fast_sentence_cbow_average_simple_hs_ptr fast_sentence_cbow_average_simple_hs
cdef fast_sentence_cbow_average_simple_neg_ptr fast_sentence_cbow_average_simple_neg

''' cbow average plus doc'''
cdef fast_sentence_cbow_average_plus_doc_hs_ptr fast_sentence_cbow_average_plus_doc_hs
cdef fast_sentence_cbow_average_plus_doc_neg_ptr fast_sentence_cbow_average_plus_doc_neg

''' cbow concat'''
cdef fast_sentence_cbow_hs_concat_ptr fast_sentence_cbow_hs_concat
cdef fast_sentence_cbow_neg_concat_ptr fast_sentence_cbow_neg_concat

''' sg simple'''
cdef fast_sentence_sg_hs_simple_ptr fast_sentence_sg_hs_simple
cdef fast_sentence_sg_neg_simple_ptr fast_sentence_sg_neg_simple

''' cbow'''
cdef fast_sentence_cbow_hs_ptr fast_sentence_cbow_hs
cdef fast_sentence_cbow_neg_ptr fast_sentence_cbow_neg

''' sg'''
cdef fast_sentence_sg_hs_ptr fast_sentence_sg_hs
cdef fast_sentence_sg_neg_ptr fast_sentence_sg_neg


DEF EXP_TABLE_SIZE = 10000
DEF MAX_EXP = 12

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

cdef void fast_sentence0_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>dsdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)


cdef void fast_sentence1_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)


cdef void fast_sentence2_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    for a in range(size):
        work[a] = <REAL_t>0.0
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>0.0
        for a in range(size):
            f += syn0[row1 + a] * syn1[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * syn0[row1 + a]
    for a in range(size):
        syn0[row1 + a] += work[a]


cdef unsigned long long fast_sentence0_sg_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>dsdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    return next_random

cdef unsigned long long fast_sentence1_sg_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    return next_random

cdef unsigned long long fast_sentence2_sg_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    for a in range(size):
        work[a] = <REAL_t>0.0

    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>0.0
        for a in range(size):
            f += syn0[row1 + a] * syn1neg[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        for a in range(size):
            work[a] += g * syn1neg[row2 + a]
        for a in range(size):
            syn1neg[row2 + a] += g * syn0[row1 + a]

    for a in range(size):
        syn0[row1 + a] += work[a]

    return next_random

cdef void fast_sentence0_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m] * size], &ONE)

cdef void fast_sentence1_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = <REAL_t>sdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

cdef void fast_sentence2_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count
    cdef int m

    for a in range(size):
        neu1[a] = <REAL_t>0.0
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            for a in range(size):
                neu1[a] += syn0[indexes[m] * size + a]
    if cbow_mean and count > (<REAL_t>0.5):
        for a in range(size):
            neu1[a] /= count

    for a in range(size):
        work[a] = <REAL_t>0.0
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = <REAL_t>0.0
        for a in range(size):
            f += neu1[a] * syn1[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * neu1[a]

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            for a in range(size):
                syn0[indexes[m] * size + a] += work[a]

cdef unsigned long long fast_sentence0_cbow_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random

cdef unsigned long long fast_sentence1_cbow_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>sdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random

cdef unsigned long long fast_sentence2_cbow_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    for a in range(size):
        neu1[a] = <REAL_t>0.0
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            for a in range(size):
                neu1[a] += syn0[indexes[m] * size + a]
    if cbow_mean and count > (<REAL_t>0.5):
        for a in range(size):
            neu1[a] /= count

    for a in range(size):
        work[a] = <REAL_t>0.0

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>0.0
        for a in range(size):
            f += neu1[a] * syn1neg[row2 + a]
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        for a in range(size):
            work[a] += g * syn1neg[row2 + a]
        for a in range(size):
            syn1neg[row2 + a] += g * neu1[a]

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            for a in range(size):
                syn0[indexes[m] * size + a] += work[a]

    return next_random


''' normal skip'''
def train_sentence_sg(model, sentence, alpha, _work):
    cdef int hs = model.hs
    cdef int negative = model.negative

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                if hs:
                    fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work)
                if negative:
                    next_random = fast_sentence_sg_neg(negative, table, table_len, syn0, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random)

    return result

''' normal cbow'''
def train_sentence_cbow(model, sentence, alpha, _work, _neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            if hs:
                fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean)
            if negative:
                next_random = fast_sentence_cbow_neg(negative, table, table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random)

    return result




# 中間層はdoc_vecとword_vecの連結
''' concat skip '''
def train_sentence_sg_concat(model, sentence_id, sentence, alpha, _work, _neu1, alpha_doc):
    cdef int hs = model.hs
    cdef int negative = model.negative

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0)) # simpleでは不要
    cdef REAL_t *doc = <REAL_t *>(np.PyArray_DATA(model.doc))

    # init l1
    cdef REAL_t *l1 = <REAL_t *>np.PyArray_DATA(_neu1)

    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef REAL_t _alpha_doc = alpha_doc
    cdef int size = model.layer1_size
    cdef int doc_size = model.doc_vec_size
    cdef int l1_size = size+doc_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    # cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window
    cdef int window_r = model.window_r

    cdef int i, j, k
    cdef long result = 0
    cdef long _sentence_id = sentence_id*doc_size
    

    cdef int cbow_mean = model.cbow_mean
    cdef int average_flag = model.average_flag

    cdef int flag_freeze_learning = model.freeze_learn

    # if sentence_id in [0,2,149066]:
        # print "sentence_id : ",sentence_id
        # print "inner af : ",doc[_sentence_id]
    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            # reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            # j = i - window + reduced_windows[i]
            j = i - window # no reduce
            if j < 0:
                j = 0
            # k = i + window_r + 1 - reduced_windows[i]
            k = i + window_r + 1 # no reduce
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                if hs:
                    fast_sentence1_sg_hs_concat(_sentence_id,points[i], codes[i], codelens[i], doc, syn0, syn1, size,doc_size, indexes[j], _alpha, work,l1,cbow_mean, average_flag,flag_freeze_learning)
                if negative:
                    next_random = fast_sentence1_sg_neg_concat(_sentence_id, negative, table, table_len, doc,syn0, syn1neg, size,doc_size, indexes[i], indexes[j], _alpha, work, next_random,l1,cbow_mean, average_flag,flag_freeze_learning)


    # if sentence_id in [0,2,149066]:
        # print "inner af : ",doc[_sentence_id]
    return result


''' concat skip hs'''
cdef void fast_sentence1_sg_hs_concat(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *doc, REAL_t *syn0, REAL_t *syn1, const int size,const int doc_size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *l1, const int cbow_mean,const int average_flag,const int flag_freeze_learning) nogil:
    
    # print sentence_id
    cdef long long a, b
    cdef int l1_size = size+doc_size
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g, inv_count
    cdef long exp_index

    memset(work, 0, l1_size * cython.sizeof(REAL_t))
    memset(l1, 0, l1_size * cython.sizeof(REAL_t))


    saxpy(&size, &ONEF, &syn0[row1], &ONE, l1, &ONE)
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, &l1[size], &ONE)

    # saxpy(&size, &ONEF, &syn0[row1], &ONE, l1, &ONE)
    # # doc add
    # saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, l1, &ONE)

    # if cbow_mean == 1 and average_flag == 0:
    #     inv_count = ONEF+ONEF
    #     sscal(&size, &inv_count, l1, &ONE)


    for b in range(codelen):
        row2 = word_point[b] * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        f = <REAL_t>sdot(&l1_size, l1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
            continue
        f = EXP_TABLE[exp_index]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&l1_size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&l1_size, &g, l1, &ONE, &syn1[row2], &ONE)


    if flag_freeze_learning == 0:
        saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    saxpy(&doc_size, &ONEF, &work[size], &ONE, &doc[sentence_id], &ONE)




''' concat skip negative'''
cdef unsigned long long fast_sentence1_sg_neg_concat(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *doc, REAL_t *syn0, REAL_t *syn1neg, const int size,const int doc_size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *l1, const int cbow_mean,const int average_flag,const int flag_freeze_learning) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label,inv_count
    cdef np.uint32_t target_index
    cdef int d

    cdef int l1_size = size+doc_size

    memset(work, 0, l1_size * cython.sizeof(REAL_t))
    memset(l1, 0, l1_size * cython.sizeof(REAL_t))

    saxpy(&size, &ONEF, &syn0[row1], &ONE, l1, &ONE)
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, &l1[size], &ONE)


    # saxpy(&size, &ONEF, &syn0[row1], &ONE, l1, &ONE)
    # # doc add
    # saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, l1, &ONE)

    # if cbow_mean == 1 and average_flag == 0:
    #     inv_count = ONEF+ONEF
    #     sscal(&size, &inv_count, l1, &ONE)



    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        f = <REAL_t>sdot(&l1_size, l1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&l1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&l1_size, &g, l1, &ONE, &syn1neg[row2], &ONE)



    if flag_freeze_learning == 0:
        saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    saxpy(&doc_size, &ONEF, &work[size], &ONE, &doc[sentence_id], &ONE)

    return next_random




# 中間層はdoc_vecとword_vecの平均
''' average skip '''
def train_sentence_sg_average(model, sentence_id, sentence, alpha, _work, _neu1, alpha_doc):
    cdef int hs = model.hs
    cdef int negative = model.negative

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0)) # simpleでは不要
    cdef REAL_t *doc = <REAL_t *>(np.PyArray_DATA(model.doc))

    # init l1
    cdef REAL_t *l1 = <REAL_t *>np.PyArray_DATA(_neu1)

    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef REAL_t _alpha_doc = alpha_doc
    cdef int size = model.layer1_size
    cdef int doc_size = model.doc_vec_size
    cdef int l1_size = size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    # cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window
    cdef int window_r = model.window_r

    cdef int i, j, k
    cdef long result = 0
    cdef long _sentence_id = sentence_id*doc_size
    

    cdef int cbow_mean = model.cbow_mean
    cdef int average_flag = model.average_flag

    cdef int flag_freeze_learning = model.freeze_learn

    # if sentence_id in [0,2,149066]:
        # print "sentence_id : ",sentence_id
        # print "inner af : ",doc[_sentence_id]
    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            # reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            # j = i - window + reduced_windows[i]
            j = i - window # no reduce
            if j < 0:
                j = 0
            # k = i + window_r + 1 - reduced_windows[i]
            k = i + window_r + 1 # no reduce
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                if hs:
                    fast_sentence1_sg_hs_average(_sentence_id,points[i], codes[i], codelens[i], doc, syn0, syn1, size,doc_size, indexes[j], _alpha, work,l1,cbow_mean, average_flag,flag_freeze_learning)
                if negative:
                    next_random = fast_sentence1_sg_neg_average(_sentence_id, negative, table, table_len, doc,syn0, syn1neg, size,doc_size, indexes[i], indexes[j], _alpha, work, next_random,l1,cbow_mean, average_flag,flag_freeze_learning)


    # if sentence_id in [0,2,149066]:
        # print "inner af : ",doc[_sentence_id]
    return result



''' average skip hs'''
cdef void fast_sentence1_sg_hs_average(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *doc, REAL_t *syn0, REAL_t *syn1, const int size,const int doc_size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *l1, const int cbow_mean,const int average_flag,const int flag_freeze_learning) nogil:
    
    # print sentence_id
    cdef long long a, b
    cdef int l1_size = size
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g, inv_count
    cdef long exp_index

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, l1_size * cython.sizeof(REAL_t))


    saxpy(&size, &ONEF, &syn0[row1], &ONE, l1, &ONE)
    # doc add
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, l1, &ONE)

    if cbow_mean == 1 and average_flag == 0:
        inv_count = ONEF+ONEF
        sscal(&size, &inv_count, l1, &ONE)


    for b in range(codelen):
        row2 = word_point[b] * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        f = <REAL_t>sdot(&l1_size, l1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
            continue
        f = EXP_TABLE[exp_index]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&l1_size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&l1_size, &g, l1, &ONE, &syn1[row2], &ONE)


    if flag_freeze_learning == 0:
        saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    saxpy(&doc_size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)




''' average skip negative'''
cdef unsigned long long fast_sentence1_sg_neg_average(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *doc, REAL_t *syn0, REAL_t *syn1neg, const int size,const int doc_size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *l1, const int cbow_mean,const int average_flag,const int flag_freeze_learning) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label,inv_count
    cdef np.uint32_t target_index
    cdef int d

    cdef int l1_size = size

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, l1_size * cython.sizeof(REAL_t))


    saxpy(&size, &ONEF, &syn0[row1], &ONE, l1, &ONE)
    # doc add
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, l1, &ONE)

    if cbow_mean == 1 and average_flag == 0:
        inv_count = ONEF+ONEF
        sscal(&size, &inv_count, l1, &ONE)



    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        f = <REAL_t>sdot(&l1_size, l1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&l1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&l1_size, &g, l1, &ONE, &syn1neg[row2], &ONE)



    if flag_freeze_learning == 0:
        saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)

    saxpy(&doc_size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)

    return next_random


''' simple skip '''
def train_sentence_sg_simple(model, sentence_id, sentence, alpha, _work, alpha_doc):
    cdef int hs = model.hs
    cdef int negative = model.negative

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0)) # simpleでは不要
    cdef REAL_t *doc = <REAL_t *>(np.PyArray_DATA(model.doc))

    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef REAL_t _alpha_doc = alpha_doc
    cdef int size = model.layer1_size
    cdef int doc_size = model.doc_vec_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    # cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window
    cdef int window_r = model.window_r

    cdef int i, j, k
    cdef long result = 0
    cdef long _sentence_id = sentence_id*doc_size
    
    # if sentence_id in [0,2,149066]:
        # print "sentence_id : ",sentence_id
        # print "inner af : ",doc[_sentence_id]
    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            # reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            # j = i - window + reduced_windows[i]
            j = i - window # no reduce
            if j < 0:
                j = 0
            # k = i + window_r + 1 - reduced_windows[i]
            k = i + window_r + 1 # no reduce
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                if hs:
                    fast_sentence_sg_hs_simple(_sentence_id,points[i], codes[i], codelens[i], doc, syn1, size, indexes[j], _alpha, work)
                if negative:
                    next_random = fast_sentence_sg_neg_simple(_sentence_id, negative, table, table_len, doc, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random)


    # if sentence_id in [0,2,149066]:
        # print "inner af : ",doc[_sentence_id]
    return result


cdef void fast_sentence0_sg_hs_simple(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *doc, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:
    
    # print sentence_id
    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g
    cdef long exp_index

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        f = <REAL_t>dsdot(&size, &doc[sentence_id], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
            continue
        f = EXP_TABLE[exp_index]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &doc[sentence_id], &ONE, &syn1[row2], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)


cdef void fast_sentence1_sg_hs_simple(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *doc, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work) nogil:
    
    # print sentence_id
    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g
    cdef long exp_index

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        f = <REAL_t>sdot(&size, &doc[sentence_id], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
            continue
        f = EXP_TABLE[exp_index]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &doc[sentence_id], &ONE, &syn1[row2], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)



''' simple skip negative'''
cdef unsigned long long fast_sentence0_sg_neg_simple(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *doc, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        f = <REAL_t>dsdot(&size, &doc[sentence_id], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &doc[sentence_id], &ONE, &syn1neg[row2], &ONE)


    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)

    return next_random

''' simple skip negative'''
cdef unsigned long long fast_sentence1_sg_neg_simple(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *doc, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        # f = <REAL_t>sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        f = <REAL_t>sdot(&size, &doc[sentence_id], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &doc[sentence_id], &ONE, &syn1neg[row2], &ONE)


    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)

    return next_random

''' concat cbow '''
def train_sentence_cbow_concatenate(model, sentence_id,  sentence, alpha, _work, _neu1, alpha_doc):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *doc = <REAL_t *>(np.PyArray_DATA(model.doc)) # add
    cdef REAL_t *null_vec = <REAL_t *>(np.PyArray_DATA(model.null_vec)) # add
    cdef REAL_t *neu1
    cdef REAL_t _alpha = alpha
    cdef REAL_t _alpha_doc = alpha_doc # add
    cdef int size = model.layer1_size
    cdef int doc_size = model.doc_vec_size # add

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    # cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN] # no reduce
    cdef int sentence_len
    cdef int window = model.window
    cdef int window_r = model.window_r # add
    cdef long word_vec_length_total = (window+window_r)*size # add
    cdef int l1_size = word_vec_length_total+doc_size
    cdef int i, j, k
    cdef long result = 0
    cdef long _sentence_id = sentence_id*doc_size # add
    cdef long sent_i
    cdef long l1_i
    cdef long l1_j,size_i
    cdef long top_i,m,bottom_i
    cdef int flag_freeze_learning = model.freeze_learn
    cdef int skip_concat = model.skip_concat # nullを含む場合にはスキップ
    # init l1
    cdef REAL_t *l1 = <REAL_t *>np.PyArray_DATA(_neu1)
    memset(l1, 0, (l1_size) * cython.sizeof(REAL_t))

    # print "init work"
    # init work
    cdef REAL_t *work = <REAL_t *>np.PyArray_DATA(_work)
    memset(work, 0, (l1_size) * cython.sizeof(REAL_t))
    cdef int iii

    # if sentence_id == 0:
    #     print "cython"
    # else:
    #     print "g_cython"
    # print "syn0"
    # print model.syn0[1]


    # print "syn0_here"
    # for iii in range(size):
    #     print syn0[iii+size]
    # print "doc"
    # print model.doc[1]


    # print "doc_here"
    # for iii in range(doc_size):
    #     print doc[iii+doc_size]

    # print "shape"
    # print model.null_vec.shape
    # print type(model.syn0)
    # print type(model.null_vec)
    # print "null_vec_py"
    # print model.null_vec


    # print "null_vec_here"
    # for iii in range(size):
    #     print null_vec[iii],
        # print null_vec+iii,

    # print "\n"
    # print "word_vec_length_total", word_vec_length_total
    # print "doc_size", doc_size
    # print "total", l1_size
    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    # print "doc_original"
    # print model.doc[0]
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:

            indexes[i] = word.index
            # reduced_windows[i] = np.random.randint(window) # no reduce
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
                # if sentence_id == 0:
                    # print word.point
            else:
                codelens[i] = 1
            result += 1

    # print "nogil"
    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            # print "\n-----------------"
            # print "i = ",i
            # print "-----------------"
            if codelens[i] == 0:
                continue
            
            # if sentence_id == 0:
                # print "points[i]  : ", points[i]

            # print "l1 : ", l1_size
            memset(work, 0, l1_size * cython.sizeof(REAL_t))
            memset(l1, 0, l1_size * cython.sizeof(REAL_t))

            # print "memset af"
            j = i - window # no reduce

            if j < 0:
                j = 0
                if skip_concat == 1:
                    continue
            # k = i + window + 1 - reduced_windows[i]
            k = i + window_r + 1  # no reduce
            if k > sentence_len:
                k = sentence_len
                if skip_concat == 1:
                    continue

            l1_i = 0

            # top
            for top_i in range(window-i):
                saxpy(&size, &ONEF, null_vec, &ONE, &l1[l1_i*size], &ONE)
                l1_i += 1


            #     print "\nSTEP TOP:"
            #     print "window-i : ",window-i
            #     print "l1_i : ",l1_i

            #     print "\n"
            #     print "\n"
            #     for iii in range(l1_size):
            #         print l1[iii],


            # indcies
            for m in range(j,k):
                if m == i or codelens[m] == 0:
                    continue

                # if sentence_id == 0:
                #     print "\nCENTER!!\n"
                saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, &l1[l1_i*size], &ONE)
                l1_i += 1



            # if sentence_id == 0:
            #     print "\nSTEP CENTER:"
            #     print "j : ",j
            #     print "k : ",k

            #     print "\n"
            #     print "\n"

            #     for iii in range(l1_size):
            #         print l1[iii],


            # bottom
            for bottom_i in range(i + window_r-sentence_len+1):
                saxpy(&size, &ONEF, null_vec, &ONE, &l1[l1_i*size], &ONE)
                l1_i += 1

            # if sentence_id == 0:
            #     print "STEP BOTTOM:"
            #     print "i + window_r-sentence_len+1 : ", i + window_r-sentence_len+1
            #     print "\n"
            #     print "\n"
            #     for iii in range(l1_size):
            #         print l1[iii],
            # print "l1_i",l1_i
            #doc
            saxpy(&doc_size, &ONEF, &doc[_sentence_id], &ONE, &l1[l1_i*size], &ONE) 


            # print "sentence_id : ", sentence_id
            # if sentence_id == 0:
            #     for iii in range(l1_size):
            #         print l1[iii],


            # if sentence_id == 0:
            #     print "\nSTEP DOC:"
            #     print "\n"
            #     print "\n"
            #     for iii in range(l1_size):
            #         print l1[iii],



            #     print "\nDOC DOC:"
            #     for iii in range(doc_size):
            #         print doc[_sentence_id+iii],

            # print "\n"
            if hs:
                fast_sentence_cbow_hs_concat(_sentence_id,points[i], codes[i], codelens,null_vec ,window , window_r,sentence_len, l1 , doc, syn0, syn1, size ,doc_size, l1_size,indexes, _alpha, work, i, j, k, cbow_mean,flag_freeze_learning)
            if negative:
                next_random = fast_sentence_cbow_neg_concat(_sentence_id,negative, table, table_len, codelens,null_vec ,window , window_r,sentence_len, l1, doc ,syn0, syn1neg, size, doc_size, l1_size, indexes, _alpha, work, i, j, k, cbow_mean, next_random,flag_freeze_learning)

    return result



''' conca cbow hs'''
cdef void fast_sentence1_cbow_hs_concat(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *null_vec,const int window, const int window_r,const int sentence_len, REAL_t *l1, REAL_t *doc, REAL_t *syn0, REAL_t *syn1, 
    const int size,const int doc_size, const int l1_size, const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, int flag_freeze_learning) nogil:# 

    cdef long _sentence_id = sentence_id# add
    cdef long long a, b
    cdef long long size_i,d_i
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef long m
    cdef long l1_i = window + window_r
    cdef long l1_j
    cdef long top_i,bottom_i
    cdef long word_start_i = 0
    cdef int exp_index
    # cdef int flag_freeze_learning_ = flag_freeze_learning

    memset(work, 0, (l1_size) * cython.sizeof(REAL_t))
    # for l1_j in range(l1_size):
    #         print "l1 or ",l1[l1_j],l1_j

    # if sentence_id == 0:
    #     print "codelens[i] : ", codelens[i]
    #     for iii in range(l1_size):
    #         print l1[iii],
    #     print ""

    for b in range(codelens[i]):
        row2 = word_point[b] * l1_size
        # print "word_point[b] : ",word_point[b]
        # print "row2 values : ", row2

        # if sentence_id == 0:
        #     print "syn1[row2]"
        #     for iii in range(l1_size):
        #         print syn1[row2+iii],
        #     print ""


        f = <REAL_t>sdot(&l1_size, l1, &ONE, &syn1[row2], &ONE)
      
        # if f <= -MAX_EXP or f >= MAX_EXP:
        #     continue
        # exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        # if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
        #     continue
        # f = <REAL_t> EXP_TABLE[exp_index]


        f = 1.0 / (1.0 + exp(-f))


        # if sentence_id == 0:
        #     print "f", f

        # original code
        # f = <REAL_t>sdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        # if f <= -MAX_EXP or f >= MAX_EXP:
        #     continue
        # f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
  

        # if sentence_id == 0:
        #     print "row2 : ",row2

        #     print "syn1 before"
        #     # print "codelens[i] : ", codelens[i]
        #     for iii in range(l1_size):
        #         print syn1[row2+iii],
        #     print ""
        #     print ""


        g = (1 - word_code[b] - f) * alpha
        saxpy(&l1_size, &g, &syn1[row2], &ONE, work, &ONE) 
        saxpy(&l1_size, &g, l1, &ONE, &syn1[row2], &ONE) 


        # if sentence_id == 0:
        #     print "syn1 after"
        #     # print "codelens[i] : ", codelens[i]
        #     for iii in range(l1_size):
        #         print syn1[row2+iii],
        #     print ""
        #     print ""

        # if sentence_id == 0:
        #     for iii in range(l1_size):
        #         print work[iii],
        #     print ""

    word_start_i = 0
    for top_i in range(window-i):
        word_start_i += 1
    # word indexes
    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue

        if flag_freeze_learning == 0:
            saxpy(&size, &ONEF, &work[word_start_i*size], &ONE, &syn0[indexes[m] * size], &ONE)
        word_start_i += 1

    # null word
    for bottom_i in range(i + window_r-sentence_len+1):
        word_start_i += 1

    # print "word_start_i*size",word_start_i*size
    # #doc
    saxpy(&doc_size, &ONEF, &work[word_start_i*size], &ONE, &doc[_sentence_id], &ONE)



''' conca cbow negative'''
cdef unsigned long long fast_sentence1_cbow_neg_concat(
    const long sentence_id, const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *null_vec,const int window, const int window_r,const int sentence_len, REAL_t *l1, REAL_t *doc, REAL_t *syn0,
    REAL_t *syn1neg, const int size,const int doc_size, const int l1_size, 
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random,int flag_freeze_learning)nogil :#nogil


    cdef long _sentence_id = sentence_id# add

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    cdef long l1_i = window + window_r
    cdef long l1_j
    cdef long top_i,bottom_i
    cdef long word_start_i = 0
    cdef int exp_index


    word_index = indexes[i]

    memset(work, 0, (l1_size) * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * l1_size
        f = <REAL_t>sdot(&l1_size, l1, &ONE, &syn1neg[row2], &ONE)
        
        f = 1.0 / (1.0 + exp(-f))

        # if f <= -MAX_EXP or f >= MAX_EXP:
        #     continue
        # exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        # if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
        #     continue
        # f = EXP_TABLE[exp_index]
        # print "f ", f
        # if sentence_id == 0:
        #     print "row2 : ",row2

        #     print "syn1 before"
        #     # print "codelens[i] : ", codelens[i]
        #     for iii in range(l1_size):
        #         print syn1neg[row2+iii],
        #     print ""
        #     print ""

        g = (label - f) * alpha
        saxpy(&l1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&l1_size, &g, l1, &ONE, &syn1neg[row2], &ONE)


        # if sentence_id == 0:
        #     print "row2 : ",row2

        #     print "syn1 after"
        #     # print "codelens[i] : ", codelens[i]
        #     for iii in range(l1_size):
        #         print syn1neg[row2+iii],
        #     print ""
        #     print ""

    word_start_i = 0
    for top_i in range(window-i):
        word_start_i += 1
    # word indexes
    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue

        if flag_freeze_learning == 0:
            saxpy(&size, &ONEF, &work[word_start_i*size], &ONE, &syn0[indexes[m] * size], &ONE)
        word_start_i += 1

    # null word
    for bottom_i in range(i + window_r-sentence_len+1):
        word_start_i += 1

    # #doc
    saxpy(&doc_size, &ONEF, &work[word_start_i*size], &ONE, &doc[_sentence_id], &ONE)

    return next_random



''' conca cbow hs'''
cdef void fast_sentence0_cbow_hs_concat(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *null_vec,const int window, const int window_r,const int sentence_len, REAL_t *l1, REAL_t *doc, REAL_t *syn0, REAL_t *syn1, 
    const int size,const int doc_size, const int l1_size, const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, int flag_freeze_learning) nogil:#

    cdef long _sentence_id = sentence_id# add
    cdef long long a, b
    cdef long long size_i,d_i
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef long m
    cdef long l1_i = window + window_r
    cdef long l1_j
    cdef long top_i,bottom_i
    cdef long word_start_i = 0
    cdef int exp_index
    # cdef int flag_freeze_learning_ = flag_freeze_learning

    memset(work, 0, (l1_size) * cython.sizeof(REAL_t))
    # for l1_j in range(l1_size):
    #         print "l1 or ",l1[l1_j],l1_j

    for b in range(codelens[i]):
        row2 = word_point[b] * l1_size
        # print "f start"

        f = <REAL_t>dsdot(&l1_size, l1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
            continue

        f = <REAL_t> EXP_TABLE[exp_index]

        g = (1 - word_code[b] - f) * alpha
        saxpy(&l1_size, &g, &syn1[row2], &ONE, work, &ONE) 
        saxpy(&l1_size, &g, l1, &ONE, &syn1[row2], &ONE) 

    word_start_i = 0
    for top_i in range(window-i):
        word_start_i += 1
    # word indexes
    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue

        if flag_freeze_learning == 0:
            saxpy(&size, &ONEF, &work[word_start_i*size], &ONE, &syn0[indexes[m] * size], &ONE)
        word_start_i += 1

    # null word
    for bottom_i in range(i + window_r-sentence_len+1):
        word_start_i += 1

    # #doc
    saxpy(&doc_size, &ONEF, &work[word_start_i*size], &ONE, &doc[_sentence_id], &ONE)



''' conca cbow negative'''
cdef unsigned long long fast_sentence0_cbow_neg_concat(
    const long sentence_id, const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *null_vec,const int window, const int window_r,const int sentence_len, REAL_t *l1, REAL_t *doc, REAL_t *syn0,
    REAL_t *syn1neg, const int size,const int doc_size, const int l1_size, 
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random,int flag_freeze_learning) nogil:



    cdef long _sentence_id = sentence_id# add

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    cdef long l1_i = window + window_r
    cdef long l1_j
    cdef long top_i,bottom_i
    cdef long word_start_i = 0
    cdef int exp_index


    word_index = indexes[i]

    memset(work, 0, (l1_size) * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * l1_size
        f = <REAL_t>dsdot(&l1_size, l1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        exp_index = <int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))
        if exp_index >= EXP_TABLE_SIZE or exp_index < 0:
            continue
        f = EXP_TABLE[exp_index]
        g = (label - f) * alpha
        saxpy(&l1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&l1_size, &g, l1, &ONE, &syn1neg[row2], &ONE)


    word_start_i = 0
    for top_i in range(window-i):
        word_start_i += 1
    # word indexes
    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue

        if flag_freeze_learning == 0:
            saxpy(&size, &ONEF, &work[word_start_i*size], &ONE, &syn0[indexes[m] * size], &ONE)
        word_start_i += 1

    # null word
    for bottom_i in range(i + window_r-sentence_len+1):
        word_start_i += 1

    # #doc
    saxpy(&doc_size, &ONEF, &work[word_start_i*size], &ONE, &doc[_sentence_id], &ONE)

    return next_random



''' cbow average'''
def train_sentence_cbow_average_simple(model, sentence_id, sentence, alpha, _work, _neu1, alpha_doc):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int cbow_mean = model.cbow_mean


    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *doc = <REAL_t *>(np.PyArray_DATA(model.doc))
    cdef REAL_t *work
    cdef REAL_t *l1
    cdef REAL_t _alpha = alpha
    cdef REAL_t _alpha_doc = alpha_doc # add
    cdef int size = model.layer1_size
    cdef int doc_size = model.doc_vec_size

    cdef long _sentence_id = sentence_id*doc_size # add
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window
    cdef int window_r = model.window_r # add

    cdef int average_flag = model.average_flag

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    l1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            # reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window # + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window_r + 1 # - reduced_windows[i]

            if k > sentence_len:
                k = sentence_len


            memset(l1, 0, size * cython.sizeof(REAL_t))
            memset(work, 0, size * cython.sizeof(REAL_t))

            if hs:
                fast_sentence_cbow_average_simple_hs(_sentence_id ,points[i], codes[i], codelens, l1, syn0, syn1, doc, size, indexes, _alpha, work, i, j, k, cbow_mean,average_flag)
            if negative:
                next_random = fast_sentence_cbow_average_simple_neg(_sentence_id,negative, table, table_len, codelens, l1, syn0, syn1neg, doc, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random,average_flag)

    return result




cdef unsigned long long fast_sentence0_cbow_average_simple_neg(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, REAL_t *doc, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, int average_flag) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    if cbow_mean == 1  and count > (<REAL_t>0.0) and average_flag == 0:
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)


    # doc add
    saxpy(&size, &ONEF, &doc[sentence_id], &ONE, neu1, &ONE)
    count += ONEF


    if cbow_mean == 1 and count > (<REAL_t>0.0) and average_flag == 1:
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)



    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        # f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        # if f <= -MAX_EXP or f >= MAX_EXP:
            # continue
        # f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        f = 1.0 / (1.0 + exp(-f))

        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)

    return next_random



cdef unsigned long long fast_sentence1_cbow_average_simple_neg(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *l1,  REAL_t *syn0, REAL_t *syn1neg, REAL_t *doc, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, int average_flag) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, l1, &ONE)


    if cbow_mean == 1  and average_flag == 0:
        inv_count = ONEF/count
        sscal(&size, &inv_count, l1, &ONE)

    # doc add
    saxpy(&size, &ONEF, &doc[sentence_id], &ONE, l1, &ONE)
    count += ONEF

    if cbow_mean == 1 and average_flag == 1:
        inv_count = ONEF/count
        sscal(&size, &inv_count, l1, &ONE)



    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        # f = <REAL_t>sdot(&size, l1, &ONE, &syn1neg[row2], &ONE)
        # if f <= -MAX_EXP or f >= MAX_EXP:
            # continue
        # f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        f = <REAL_t>sdot(&size, l1, &ONE, &syn1neg[row2], &ONE)
        f = 1.0 / (1.0 + exp(-f))

        g = (label - f) * alpha
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, l1, &ONE, &syn1neg[row2], &ONE)

    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)

    return next_random




cdef void fast_sentence1_cbow_average_simple_hs(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *l1, REAL_t *syn0, REAL_t *syn1, REAL_t *doc, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, int average_flag) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, l1, &ONE)


    if cbow_mean == 1 and average_flag == 0:
        inv_count = ONEF/count
        sscal(&size, &inv_count, l1, &ONE)

    # doc add
    saxpy(&size, &ONEF, &doc[sentence_id], &ONE, l1, &ONE)
    count += ONEF

    if cbow_mean == 1 and average_flag == 1:
        inv_count = ONEF/count
        sscal(&size, &inv_count, l1, &ONE)


    for b in range(codelens[i]):
        row2 = word_point[b] * size
        # f = <REAL_t>sdot(&size, l1, &ONE, &syn1[row2], &ONE)
        # if f <= -MAX_EXP or f >= MAX_EXP:
            # continue
        # f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        f = <REAL_t>sdot(&size, l1, &ONE, &syn1[row2], &ONE)
        f = 1.0 / (1.0 + exp(-f))

        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, l1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)
    # add doc
    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)



cdef void fast_sentence0_cbow_average_simple_hs(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, REAL_t *doc, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, int average_flag) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    if cbow_mean == 1 and count > (<REAL_t>0.0) and average_flag == 0:
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    # doc add
    saxpy(&size, &ONEF, &doc[sentence_id], &ONE, neu1, &ONE)
    count += ONEF

    if cbow_mean == 1 and count > (<REAL_t>0.0) and average_flag == 1:
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        # f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        # if f <= -MAX_EXP or f >= MAX_EXP:
            # continue
        # f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        f = 1.0 / (1.0 + exp(-f))

        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)
    # add doc
    saxpy(&size, &ONEF, work, &ONE, &doc[sentence_id], &ONE)





''' cbow average plus doc'''
def train_sentence_cbow_average_plus_doc(model, sentence_id, sentence, alpha, _work, _neu1, alpha_doc):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int cbow_mean = model.cbow_mean


    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *doc = <REAL_t *>(np.PyArray_DATA(model.doc))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef REAL_t _alpha = alpha
    cdef REAL_t _alpha_doc = alpha_doc
    cdef int size = model.layer1_size
    cdef int doc_size = model.doc_vec_size # add
    cdef long _sentence_id = sentence_id*doc_size# add

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            if hs:
                fast_sentence_cbow_average_plus_doc_hs(_sentence_id ,points[i], codes[i], codelens, neu1, syn0, syn1, doc, size, doc_size,indexes, _alpha, work, i, j, k, cbow_mean)
            if negative:
                next_random = fast_sentence_cbow_average_plus_doc_neg(_sentence_id,negative, table, table_len, codelens, neu1, syn0, syn1neg, doc, size,doc_size, indexes, _alpha, work, i, j, k, cbow_mean, next_random)

    return result



cdef void fast_sentence0_cbow_average_plus_doc_hs(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, REAL_t *doc, const int size, const int doc_size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:#nogil

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m
    cdef int neu1_size = size+doc_size
    cdef int iii
    memset(neu1, 0, neu1_size * cython.sizeof(REAL_t))

    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)
    # doc add
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, &neu1[size], &ONE) 
    memset(work, 0, neu1_size * cython.sizeof(REAL_t))

    for b in range(codelens[i]):
        row2 = word_point[b] * neu1_size
        f = <REAL_t>dsdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&neu1_size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)
    # add doc
    saxpy(&doc_size, &ONEF, &work[size], &ONE, &doc[sentence_id], &ONE)


cdef void fast_sentence1_cbow_average_plus_doc_hs(
    const long sentence_id,const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, REAL_t *doc, const int size, const int doc_size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:#nogil

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m
    cdef int neu1_size = size+doc_size
    cdef int iii
    memset(neu1, 0, neu1_size * cython.sizeof(REAL_t))

    # print "\n"

    # for iii in range(neu1_size):
    #     print neu1[iii],

    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)


    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    # doc add
    # saxpy(&size, &ONEF, &doc[sentence_id], &ONE, neu1, &ONE)
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, &neu1[size], &ONE) 
    # count += ONEF
    # print "\nNEW\n"
    # for iii in range(neu1_size):
    #     print neu1[iii],
    # print "\n"

    memset(work, 0, neu1_size * cython.sizeof(REAL_t))

    # print "\nNEW\n"
    # for iii in range(neu1_size):
    #     print neu1[iii],
    # print "\n"

    for b in range(codelens[i]):
        row2 = word_point[b] * neu1_size
        f = <REAL_t>sdot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&neu1_size, &g, &syn1[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)


    # print "\n work \n"
    # for iii in range(neu1_size):
    #     print work[iii],
    # print "\n"

    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)
    # add doc
    saxpy(&doc_size, &ONEF, &work[size], &ONE, &doc[sentence_id], &ONE)


cdef unsigned long long fast_sentence0_cbow_average_plus_doc_neg(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, REAL_t *doc, const int size, const int doc_size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m
    cdef int neu1_size = size+doc_size

    word_index = indexes[i]

    memset(neu1, 0, neu1_size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    # doc add
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, &neu1[size], &ONE) 
    # saxpy(&size, &ONEF, &doc[sentence_id], &ONE, neu1, &ONE)
    # count += ONEF



    memset(work, 0, neu1_size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * neu1_size
        f = <REAL_t>dsdot(&neu1_size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&neu1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&neu1_size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    saxpy(&doc_size, &ONEF, &work[size], &ONE, &doc[sentence_id], &ONE)

    return next_random


cdef unsigned long long fast_sentence1_cbow_average_plus_doc_neg(
    const long sentence_id,const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, REAL_t *doc, const int size, const int doc_size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random) nogil:
    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m
    cdef int neu1_size = size+doc_size

    word_index = indexes[i]

    memset(neu1, 0, neu1_size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    if cbow_mean and count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&size, &inv_count, neu1, &ONE)

    # doc add
    saxpy(&doc_size, &ONEF, &doc[sentence_id], &ONE, &neu1[size], &ONE) 
    # saxpy(&size, &ONEF, &doc[sentence_id], &ONE, neu1, &ONE)
    # count += ONEF



    memset(work, 0, neu1_size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * neu1_size
        f = <REAL_t>sdot(&neu1_size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        saxpy(&neu1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&neu1_size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    for m in range(j,k):
        if m == i or codelens[m] == 0:
            continue
        else:
            saxpy(&size, &ONEF, work, &ONE, &syn0[indexes[m]*size], &ONE)

    saxpy(&doc_size, &ONEF, &work[size], &ONE, &doc[sentence_id], &ONE)

    return next_random

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """

    global fast_sentence_sg_hs
    global fast_sentence_sg_neg
    global fast_sentence_cbow_hs
    global fast_sentence_cbow_neg
    global fast_sentence_sg_hs_simple
    global fast_sentence_sg_neg_simple
    global fast_sentence_cbow_hs_concat
    global fast_sentence_cbow_neg_concat
    global fast_sentence_cbow_average_simple_hs
    global fast_sentence_cbow_average_simple_neg
    global fast_sentence_cbow_average_plus_doc_hs
    global fast_sentence_cbow_average_plus_doc_neg

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        fast_sentence_sg_hs = fast_sentence0_sg_hs
        fast_sentence_sg_neg = fast_sentence0_sg_neg
        fast_sentence_cbow_hs = fast_sentence0_cbow_hs
        fast_sentence_cbow_neg = fast_sentence0_cbow_neg


        '''sg simple'''
        fast_sentence_sg_hs_simple = fast_sentence0_sg_hs_simple
        fast_sentence_sg_neg_simple = fast_sentence0_sg_neg_simple

        ''' cbow average'''
        fast_sentence_cbow_average_simple_hs = fast_sentence0_cbow_average_simple_hs
        fast_sentence_cbow_average_simple_neg = fast_sentence0_cbow_average_simple_neg

        ''' cbow average plus doc'''
        fast_sentence_cbow_average_plus_doc_hs = fast_sentence0_cbow_average_plus_doc_hs
        fast_sentence_cbow_average_plus_doc_neg = fast_sentence0_cbow_average_plus_doc_neg

        ''' cbow concat'''
        fast_sentence_cbow_hs_concat = fast_sentence0_cbow_hs_concat
        fast_sentence_cbow_neg_concat = fast_sentence0_cbow_neg_concat
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_sentence_sg_hs = fast_sentence1_sg_hs
        fast_sentence_sg_neg = fast_sentence1_sg_neg
        fast_sentence_cbow_hs = fast_sentence1_cbow_hs
        fast_sentence_cbow_neg = fast_sentence1_cbow_neg

        '''sg simple'''
        fast_sentence_sg_hs_simple = fast_sentence1_sg_hs_simple
        fast_sentence_sg_neg_simple = fast_sentence1_sg_neg_simple

        ''' cbow average'''
        fast_sentence_cbow_average_simple_hs = fast_sentence1_cbow_average_simple_hs
        fast_sentence_cbow_average_simple_neg = fast_sentence1_cbow_average_simple_neg

        ''' cbow average plus doc'''
        fast_sentence_cbow_average_plus_doc_hs = fast_sentence1_cbow_average_plus_doc_hs
        fast_sentence_cbow_average_plus_doc_neg = fast_sentence1_cbow_average_plus_doc_neg

        ''' cbow concat'''
        fast_sentence_cbow_hs_concat = fast_sentence1_cbow_hs_concat
        fast_sentence_cbow_neg_concat = fast_sentence1_cbow_neg_concat

        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        fast_sentence_sg_hs = fast_sentence2_sg_hs
        fast_sentence_sg_neg = fast_sentence2_sg_neg
        fast_sentence_cbow_hs = fast_sentence2_cbow_hs
        fast_sentence_cbow_neg = fast_sentence2_cbow_neg
        return 2

FAST_VERSION = init()  # initialize the module
print "FAST_VERSION : ",FAST_VERSION
test_flag = "test is is"
