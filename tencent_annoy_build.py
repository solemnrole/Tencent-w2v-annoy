# -*- coding:utf-8 -*-
'''
腾讯词向量地址：https://ai.tencent.com/ailab/nlp/en/index.html
              https://mp.weixin.qq.com/s/b9NWR0F7GQLYtgGSL50gQw
https://www.52nlp.cn/%E8%85%BE%E8%AE%AF%E8%AF%8D%E5%90%91%E9%87%8F%E5%AE%9E%E6%88%98-%E9%80%9A%E8%BF%87annoy%E8%BF%9B%E8%A1%8C%E7%B4%A2%E5%BC%95%E5%92%8C%E5%BF%AB%E9%80%9F%E6%9F%A5%E8%AF%A2
'''
from gensim.models import KeyedVectors
import json
from collections import OrderedDict

tc_wv_model=KeyedVectors.load_word2vec_format('Tencent_AILab_ChineseEmbedding.txt',binary=False)
word_index = OrderedDict()
for counter, key in enumerate(tc_wv_model.vocab.keys()):
    word_index[key] = counter
with open('tc_word_index.json', 'w') as fp:
    json.dump(word_index, fp)

# 开始基于腾讯词向量构建Annoy索引，腾讯词向量大概是882万条
from annoy import AnnoyIndex
# 腾讯词向量的维度是200
tc_index = AnnoyIndex(200)
i=0
for key in tc_wv_model.vocab.keys():
    v = tc_wv_model[key]
    tc_index.add_item(i, v)
    i += 1

# 这个构建时间也比较长，另外n_trees这个参数很关键，官方文档是这样说的：
# n_trees is provided during build time and affects the build time and the index size.
# A larger value will give more accurate results, but larger indexes.
# 这里首次使用没啥经验，按文档里的是10设置，到此整个流程的内存占用大概是30G左右
tc_index.build(10)
tc_index.save('tc_index_build10.index')
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



#加载使用
def load_tc_emb():
    with open('$tc_word_index_path') as fp:
        word_index = json.load(fp)
    tc_index = AnnoyIndex(200, metric='angular')
    tc_index.load('$tc_index_embedding_path')  # 加载index完毕。
    index_word = dict([(value, key) for (key, value) in word_index.items()])
    return (tc_index,word_index,index_word)
