# -*- coding: utf-8 -*-
#
from gensim import corpora, models, similarities
import faiss
from faiss import normalize_L2
import numpy as np
def simple_one():
    '''
    use IndexFlatL2
    :return:
    '''
    d = 64                           # dimension
    nb = 100000                      # database size
    nq = 10000                       # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    print(xq[0, 0])
    print(xq[0, 1])


                       # make faiss available
    index = faiss.IndexFlatL2(d)   # build the index

    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)


    k=5
    D, I = index.search(xq, k)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])

def te():

    '''

    use normalize_L2 and IndexFlatL2

    :return:
    '''
    d = 200                           # dimension
    nb = 3000000                      # database size
    training_vectors= np.random.random((nb, d)).astype('float32')*10
    print(training_vectors)
    print('normalize_L2')
    normalize_L2(training_vectors)

    print(training_vectors)
    print('IndexFlatIP')
    index=faiss.IndexFlatL2(d)

    print(index)
    print('train')
    print(index.is_trained)


    print('add')
    print(index)
    index.add(training_vectors)

    # search_vectors=index.add(normalize_L2(index_vectors))
    print('search')
    D, I =index.search(training_vectors[:100], 5)

    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])
# te()
import time
def cosine_similar():
    '''
    cosine_similarity
    use

    :return:
    '''
    d = 64                           # dimension
    nb = 105                    # database size
    np.random.seed(1234)             # make reproducible
    training_vectors= np.random.random((nb, d)).astype('float32')*10

    print('just  compare with skearn')
    from sklearn.metrics.pairwise import cosine_similarity
    ag=cosine_similarity(training_vectors)
    skearn_D=np.sort(ag,axis=1)
    print('skean result')
    print(skearn_D[:5,-6:])
    skearn_I=np.argsort(ag, axis=1)
    print(skearn_I[:5, -6:])


    print('normalize_L2')
    normalize_L2(training_vectors)
    print(training_vectors)
    index=faiss.IndexFlatIP(d)
    index.train(training_vectors)
    print('train')
    print(index.is_trained)


    print('add')
    print(index)
    index.add(training_vectors)


    print('search')
    D, I =index.search(training_vectors[:100], 5)


    print('faiss result')
    print(D[:5])  # neighbors of the 5 first queries
    print(I[:5])                   # neighbors of the 5 first queries


def cosine_similar_2():
    '''
    cosine_similarity
    use

    :return:
    '''
    d = 64                           # dimension
    nb = 100005                    # database size
    np.random.seed(1234)             # make reproducible
    training_vectors= np.random.random((nb, d)).astype('float32')*10

    # print('just  compare with skearn')
    # from sklearn.metrics.pairwise import cosine_similarity
    # ag=cosine_similarity(training_vectors)
    # skearn_D=np.sort(ag,axis=1)
    # print('skean result')
    # print(skearn_D[:5,-6:])
    # skearn_I=np.argsort(ag, axis=1)
    # print(skearn_I[:5, -6:])


    print('normalize_L2')
    normalize_L2(training_vectors)
    print(training_vectors)
    index=faiss.IndexFlatIP(d)
    index.train(training_vectors)
    print('train')
    print(index.is_trained)


    print('add')
    print(index)
    index.add(training_vectors)


    print('search')
    t1 = time.time()
    D, I =index.search(training_vectors[:100], 50)
    t2 = time.time()


    print('faiss result {}'.format(t2-t1))
    print(D[:2])  # neighbors of the 5 first queries
    print(I[:2])                   # neighbors of the 5 first queries



def IndexIVFFlat():
    d = 200                           # dimension
    nb =200*100000           # database size
    np.random.seed(1234)             # make reproducible
    training_vectors= np.random.random((nb, d)).astype('float32')*10
    print(training_vectors.shape)
    print(training_vectors.nbytes)
    normalize_L2(training_vectors)

    nlist = 10000  # 聚类中心的个数
    k = 50
    quantizer = faiss.IndexFlatIP(d)  # the other index

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # here we specify METRIC_L2, by default it performs inner-product search
    assert not index.is_trained
    index.train(training_vectors)
    assert index.is_trained
    index.nprobe = 3  # default nprobe is 1, try a few more
    index.add(training_vectors)  # add may be a bit slower as well
    t1=time.time()
    D, I = index.search(training_vectors[:100], k)  # actual search
    t2 = time.time()

    print(D[:2])  # neighbors of the 5 first queries
    print(I[:2])                   # neighbors of the 5 first queries

    print('faiss kmeans result times {}'.format(t2 - t1))


cosine_similar_2()
#IndexIVFFlat()