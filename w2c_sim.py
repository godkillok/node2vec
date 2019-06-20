# -*- coding: UTF-8 -*-
import numpy as np
try:
    import faiss
    from faiss import normalize_L2
except Exception as e:
    print(e)
import os
import datetime

import time
import logging
dt = datetime.datetime.today() + datetime.timedelta(days=-1)
dt = dt.strftime('%Y%m%d')
SENTENCE_EMBEDDING='/data/tanggp/tmp/w2vec_embed'
tag_dic_path='/data/tanggp/tmp/w2vec_fast_id'
FASTTEXT_SEARCH='/data/tanggp/tmp/sim_w2v/{}/'.format(dt)
# SENTENCE_EMBEDDING='/data/chenzk/LDA/sims/'
# FASTTEXT_SEARCH='/data/tanggp/fat1/{}/'.format(dt)
CANDY=100  #候选集
SIM_STEP=10000 #计算相似度的
FILE_STEP=300000 #上传到s3的大小 180k左右
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

BUCKET_PREFIX_INFO="short_video_nlp/short_video_vector/fasttext_recom/{}/".format(dt)


def create_folder():
    if FASTTEXT_SEARCH != '':
        try:
            os.makedirs(FASTTEXT_SEARCH)
        except :
            pass

def cosine_index(training_vectors):
    '''
    cosine_similarity exact mode
    use
    :return:
    '''
    print(training_vectors.shape)
    d = training_vectors.shape[1]                           # dimension
    # nb = 100                  # database size
    # training_vectors= np.random.random((nb, d)).astype('float32')*10
    t1 = time.time()
    normalize_L2(training_vectors)
    index=faiss.IndexFlatIP(d)
    index.train(training_vectors)

    index.add(training_vectors)
    t2 = time.time()
    logging.info('{} times is {}'.format('add and train', t2 - t1))
    return  index

def cosine_index_ivf(training_vectors):

    '''
    cosine_similarity exact mode
    use
    :return:
    '''
    print(training_vectors.shape)
    (num, d) = training_vectors.shape                      # dimension
    # nb = 100                  # database size
    # training_vectors= np.random.random((nb, d)).astype('float32')*10
    t1 = time.time()

    nlist = max(5,int(num/500))  # 聚类中心的个数
    normalize_L2(training_vectors)
    quantizer=faiss.IndexFlatIP(d)

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(training_vectors)
    index.nprobe =max(1,int(nlist*0.7))   # default nprobe is 1, try a few more
    index.add(training_vectors)
    t2 = time.time()
    logging.info('{} times is {}'.format('add and train', t2 - t1))

    quantizer.this.disown()
    index.own_fields = True

    return  index

#
# def cosine_index_(training_vectors):ke

def cosine_search(index,start,end,training_vectors,item_id):
    t1=time.time()
    end=min(training_vectors.shape[0],end)
    print('ee')
    score, sim_id=index.search(training_vectors[start:end], CANDY)
    t2=time.time()
    logging.info('{}-{} times is {} ,len of sim id is {}'.format(start,end,t2-t1,sim_id.shape))
    sim_video_id=[]
    (si, sj) = sim_id.shape
    for i in range(si):
        i_list=[]
        for j in range(sj):
            i_list.append(item_id.get(sim_id[i][j],'')+'___'+str(score[i][j]))
        sim_video_id.append(i_list)
    files_name=write_local(sim_video_id, start)
    logging.info('{}-{} has files {} up to s3  success '.format(start, end,len(files_name)))

def write_local(sim_video_id,start):
    si=len(sim_video_id)
    files_name=[]
    for i in range(0,si,FILE_STEP):
        path=FASTTEXT_SEARCH + str(start) + '_' +str(i)
        files_name.append(path)
        with open(path,'w') as f:
            for si in sim_video_id[i:i+FILE_STEP]:
                f.writelines(si[0]+'\t__\x01'+','.join(si[1:])+'\n')
    return files_name

def read_local():
    local_path=[]
    # for root, dirs, files in os.walk(SENTENCE_EMBEDDING):
    #     for file in files:
    #         local_path.append(os.path.join(root, file))
    local_path = [SENTENCE_EMBEDDING]
    item_id={}
    sentence_embedding=[]
    count=0
    with open(tag_dic_path, 'r') as f:
        lines = f.readlines()
    tag_dic={}
    for li in lines:
        li=li.strip()
        vid=li.split('\t')[-1]
        name=' '.join(li.split('\t')[0:-1])
        tag_dic[vid]=name

    test=[]
    in_node2vec=[]
    for l in local_path:
        
        with open(l,'r') as f:
            lines=f.readlines()
        for i,line in enumerate(lines):
            line=line.strip()
            vid=str(i)
            item_id[i]=tag_dic[vid]
            sentence_embedding.append(line.split(','))
            in_node2vec.append(tag_dic[vid])
            if tag_dic[vid] in ["tamil movies","movies",'tamil movie','tamil',"movie","song","songs"]:
                print(tag_dic[vid])
                test.append(line.split(','))

    test_npa=np.array(test, dtype=np.float32)

    from sklearn.metrics.pairwise import cosine_similarity
    ag=cosine_similarity(test_npa)
    print(ag) #1
    senten_npa=np.array(sentence_embedding, dtype=np.float32)

    return item_id,senten_npa

def fasttext_embedding_sim():
    t1=time.time()
    create_folder()
    item_id, sentence_embedding=read_local()
    t2=time.time()
    lines=sentence_embedding.shape[0]
    logging.info('{} times is {}'.format('read data', t2 - t1))
    index=cosine_index(sentence_embedding) #精确模式
    #index =cosine_index_ivf(sentence_embedding) #倒排模式
    t1=time.time()
    for start in range(0,lines,SIM_STEP):
            cosine_search(index, start, start+SIM_STEP, sentence_embedding, item_id)
    t2 = time.time()
    logging.info(' total cost {} '.format(t2-t1))

fasttext_embedding_sim()