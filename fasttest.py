import os
import json
from collections import defaultdict
corpus_folder="/data/tanggp/all_text_data_text_pipe/eval_golden"
tags_dic=defaultdict(int)
co=0
node=defaultdict(int)
tags_text=[]
for root, _, files in os.walk(corpus_folder):
    for file in files:
        raw_corpus_file_path = os.path.join(root, file)
        with open(raw_corpus_file_path,"r",encoding="utf8") as f:
            lines=f.readlines()
        for li in lines:
            li=json.loads(li)
            tags=li.get("tags",[])
            tags=[ta.lower().strip() for ta in tags]
            tags=sorted(tags)

            tags_text.append(' ;'.join(tags))

with open("/data/tanggp/tmp/tags_text",'w',encoding="utf8") as f:
    for ta in tags_text:
        f.writelines(ta+'\n')


import fastText as ft
FASTTEXT_SOFTWARE = '/data/tanggp/fastText-0.1.0'
#os.system("cd {} && ./fasttext skipgram -input /data/tanggp/tmp/tags_text -dim 100 -output /data/tanggp/tmp/tags_w2v".format(FASTTEXT_SOFTWARE))
FAST_TEXT_MODEL_PATH='/data/tanggp/tmp/tags_w2v.bin'
model = ft.load_model(FAST_TEXT_MODEL_PATH)
sentor_vetor_list=[]

with open("/data/tanggp/tmp/in_node2vec", "r", encoding="utf8") as f:
    in_node2vec=f.readlines()
    for text in in_node2vec:
        try:
            text=text.strip()
            sentor_vetor_array=model.get_sentence_vector(text)
            sentor_vetor = ','.join([str(w) for w in list(sentor_vetor_array)])
            sentor_vetor_list.append(sentor_vetor)
        except Exception as e:
            print("wrong text ---{}".format(text))
            print(e)

with open("/data/tanggp/tmp/w2vec_fast_id", "w", encoding="utf8") as f:
    for i,text in enumerate(in_node2vec):
        text=text.strip()
        f.writelines(text+'\t'+str(i) + '\n')

with open("/data/tanggp/tmp/w2vec_embed", "w", encoding="utf8") as f:
    for i in sentor_vetor_list:
        f.writelines(i+'\n')


