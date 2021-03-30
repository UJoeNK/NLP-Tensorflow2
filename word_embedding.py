#!/usr/bin/env python
# coding: utf-8

# # 使用数字代替文本 

# ## one-hot向量

# 使用one-hot向量十分低效，one-hot向量十分稀疏，大部分都是0
# 并且随着词汇表大小的增加，one-hot向量的维度也会不断变大

# ## 使用单独的数字进行编码

# 我们对每一个单词进行编码，每一个单词对应一个数字，这样虽然十分高效，但是：
# 1. 使用一个数字编码一个单词过于绝对，这样不能获得到单词间的联系
# 2. 使用一个数字是的模型对于该单词解释过于困难。如线性分类器为每个特征学习单个权重。因为两个单词之间的相似性与其编码之间的相似性没有任何关系，这种特征权重的组合是没有意义的

# ## word embedding

# 词嵌入为我们提供了一种使用高效、密集表示的方法，其中相似的单词具有相似的编码。重要的是，我们不必手工指定这种编码。嵌入是浮点值的密集向量（向量的长度是您指定的参数）。它们不是手动指定嵌入的值，而是可训练的参数（训练期间模型学习的权重，与模型学习稠密层权重的方式相同）。在处理大型数据集时，常见的词嵌入是8维的（对于小型数据集），最多可达1024维。更高维度的嵌入可以捕获单词之间的细粒度关系，但需要更多的数据来学习。

# In[13]:


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# 对于此数据集，进行一点说明，只是我自己的想法，此文本编码器将可逆地对任何字符串进行编码，必要时返回字节编码。  
# 有些词汇并不在我们最终训练的词汇表中。
# 最终的词汇表里面有写并不是一个完整的单词。
# 可能是 因为词汇表只有8185.可能有些词语没有收录，就只能对该单词部分字节编码了（词汇表中包含了用于构建一些大单词的部分词根，所以可以构建任何单词。）  
# 比如词汇表中没有tensorflow，就会将其拆分为单词表中有的'ten'+'sor'+'fl'+'ow'这四部分词根

# In[14]:


(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)


# The "_" in the vocabulary represent spaces. Note how the vocabulary includes whole words (ending with "_") and partial words which it can use to build larger words:
# 单词后面的_代表的是空格，词汇表中还会看到一些部分单词，这代表的是用于构建一些单词的部分词根，如‘br’，‘er’等

# In[15]:


encoder=info.features['text'].encoder
encoder.subwords[:20]


# In[16]:


for i in train_data.take(5):
    print(i)


# In[17]:


# 使用padded_batch方法来标准化评论的长度
padded_shapes=([None],())
train_batchs=train_data.shuffle(1000).padded_batch(10,padded_shapes=padded_shapes)
test_batchs=test_data.shuffle(1000).padded_batch(10,padded_shapes=padded_shapes)


# In[18]:


# 样本都被padding到了每一批中最长的样本长度
train_batch,train_label=next(iter(train_batchs))
train_batch.numpy(),train_batch.numpy().shape


# ### 创建连续词袋模型 

# 1. 接下来，嵌入层采用整数编码的词汇表，并查找每个单词索引的嵌入向量。这些向量作为模型训练来学习。嵌入向量向输出数组添加一个维度。生成的维度是：（批处理、序列、嵌入）。 
# 2. 接下来，GlobalAveragePooling1D层通过在序列维度上求平均值来为每个示例返回固定长度的输出向量。这允许模型以最简单的方式处理可变长度的输入。
# 3. 这个固定长度的输出向量通过一个具有16个隐藏单元的完全连接（密集）层进行管道传输。
# 4. 最后一层与单个输出节点紧密连接。使用sigmoid激活函数，这个值是介于0和1之间的一个浮点数，表示评审为正的概率（或置信水平）。

# In[22]:


embedding_dim=256
model=keras.Sequential([
    layers.Embedding(encoder.vocab_size,embedding_dim),
    layers.GlobalAveragePooling1D(),
    # 删除这一层，模型更加简单，获得的词向量更具有解释性
#     layers.Dense(16,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
model.summary()


# In[23]:


checkpoint_save_path = "./word_embedding/word_embedding.ckpt"

# 用于加载模型，checkpoint中保存好了已经训练好了的模型权重，加载之后可以不用训练了
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=2)


# In[24]:


model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
history=model.fit(train_batchs,validation_data=test_batchs,epochs=8,callbacks=[cp_callback])


# In[25]:


import pandas as pd
hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
hist.tail()


# In[29]:


import matplotlib.pyplot as plt

def plot_histrory(hist):
    acc=hist['accuracy']
    val_acc=hist['val_accuracy']
    loss=hist['loss']
    val_loss=hist['val_loss']
    epochs=hist['epoch']
    
    plt.figure(figsize=(12,9))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs,loss,'b',label='loss')
    plt.plot(epochs,val_loss,'y',label='val_loss')
    plt.legend()
    
    plt.figure(figsize=(12,9))
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.plot(epochs,acc,'b',label='accuracy')
    plt.plot(epochs,val_acc,'y',label='val_accuracy')
    plt.ylim((0.5,1))
    plt.legend()
    # plt.show()
    
plot_histrory(hist)
    


# 过拟合了，应该使用early stop

# ## 取回训练后的词向量

# In[30]:


# 词向量的维度是（vocab_size,embeddind_dim)
embedding=model.layers[0]
weights=embedding.get_weights()[0]
print(weights.shape)
# # shape: (vocab_size, embedding_dim)


# In[31]:


weights


# In[32]:


# 将得到的词向量与对于单词保存到文本中
# 以下代码来自于官方教程，但是我认为这是错误的
# 以至于之后我找最后最接近的10个单词结果很诡异
# 词汇表中第一个单词是'the_'
# 时间上这个单词对应的编码是[181, 8024]
# 单词the对应的是 181，‘_'对应8024
# 这样也就可以解释subwords只有7928个，voabsize是8025
# 当然，以上只是我个人想法，也许我没有理解官方教程的意思

vecs=[]
import io
encoder=info.features['text'].encoder
out_v=io.open('word_embedding/vecs.tsv','w',encoding='utf-8')
out_m=io.open('word_embedding/meta.tsv','w',encoding='utf-8')
for num, word in enumerate(encoder.subwords):
#     vec = weights[num+1] # skip 0, it's padding.
    # 注释掉的是官方教程中的,下面是我认为正确的
    vec=weights[encoder.encode(word)]
    if len(vec)==1:
        vecs.append(vec)
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()


# ## 找最相似的10个单词

# In[33]:


len(encoder.subwords),encoder.vocab_size #encoder中遍历单词个数与vocab_size不一致
# 原因是 subwords包含的是单词以及部分词根，而词汇表中包含了一些符号，如'_',','等等


# In[34]:


import numpy as np


# In[35]:


# 计算两个词向量的距离
# 此处使用的欧式距离
def calDistance(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))


# In[36]:


#  构建距离矩阵
def build_distance_matrix(vec):
    # 距离矩阵
    D=[[0 for _ in range(len(vec))] for _ in range(len(vec))]
    # 注意矩阵是关于对角线对称的，只需要计算上三角
    for i in range(len(vec)):
        if i%100==0:
            print('*',end=' ')
        for j in range(i,len(vec)):
            dis=calDistance(vec[i],vec[j])
            D[i][j]=dis
            D[j][i]=dis
    return D


# In[37]:


# 获得单词与其对应的索引
def get_index_and_word(encoder):
    vocabulary=[]
    for word in encoder.subwords:
        if word[-1]=='_' or word[-1]==' ':
            word=word[:-1] #去掉最后的‘_'
        # 这个index才是单词在embedding层中真正对应的index
        index=encoder.encode(word)
        vocabulary.append((index,word))
    return vocabulary


# In[38]:


# 包含了（index，word）其中index是embeding层中对应的index
vocab=get_index_and_word(encoder)


# In[39]:


index,word=vocab[0]
print(index)
print(word)


# In[40]:


vec=[]
words=[]
# 下面依据index构建词向量矩阵与单词矩阵
for (index,word) in vocab:
    
    #  注意有些词是合成的，所以有多个index，对于这些词，就不管了
    if len(index)==1 and word not in words:
        vec.append(weights[index[0]])
        words.append(word)
print(len(vec))


# In[41]:


words.index('good'),words.index('bad'),words.index('nice')


# In[43]:


# 测试一下 good 与 nice，应该很接近
# good与bad很远
# 欧式距离
print(f'distance between {words[68]} and {words[121]} is {calDistance(vec[68],vec[121])})')
print(f'distance between {words[68]} and {words[355]} is {calDistance(vec[68],vec[355])})')


# In[44]:


D=build_distance_matrix(vec)


# In[45]:



test_word='wonderful'
test_index=words.index(test_word)
test_index,vec[test_index]


# In[46]:


def find_top_10(D,index):
    dis=D[index] # 获得了测试单词词向量到所有单词词向量的距离
    
    #下面我们需要找到distances中最小的k个元素，并且返回他们的索引就可以了
    # argsort函数返回的是数组值从小到大的索引值
    # 将distances变为np.array，然后对索引排序，取前k个
    minK=np.argsort(np.array(dis))[:10]
    print(minK)
    result=[]
    for i in range(10):
        result.append(words[minK[i]])
    return result


# In[47]:


# 每次的结果都不一定，与训练的epoch，embedding层的初始化有关
find_top_10(D,test_index)


# ## 读取Glove词向量

# In[1]:


import numpy as np


# In[2]:


# 因为github上传100MB以上的大文件很麻烦
# 所以我在语料库中放了glove的压缩文件，
# 解压后使用
with open('语料库/glove.6B.50d.txt') as f:
    words_set=set()
    word2vec={}
    for line in f:
        line=line.strip().split()
        cur_word=line[0]
        words_set.add(cur_word)
        word2vec[cur_word]=np.array(line[1:],dtype=np.float64)


# ## 计算单词的余弦相似度

# $$\text{CosineSimilarity(u, v)} = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta) \tag{1}$$
# 
# <img src="pic/cosine_sim.png" style="width:800px;height:250px;">
# <caption><center> **Figure 1**: The cosine of the angle between two vectors is a measure of how similar they are</center></caption>

# In[3]:


def cos_similaryty(u,v):
    dot=np.dot(u,v)
    
    norm_u=np.sqrt(np.sum(np.power(u,2)))
    norm_v=np.sqrt(np.sum(np.power(v,2)))
    
    return np.divide(dot,(norm_u*norm_v))


# In[7]:


man=word2vec['man']
woman=word2vec['woman']

computer=word2vec['computer']
basketball=word2vec['basketball']
football=word2vec['football']

print('cos similaryty between man and women is:',cos_similaryty(man,woman))
print('cos similaryty between computer and basketnall is:',cos_similaryty(computer,basketball))
print('cos similaryty between football and basketball is:',cos_similaryty(football,basketball))


# ## 计算相似的词

# a to b like c to __ 
# 例如:
# - man to women like king to quuen
# - Paris to France like Beijing to China

# In[13]:


def caculate_similar(a,b,c,word2vec):
    
    a,b,c=a.lower(),b.lower(),c.lower()
    vec_a,vec_b,vec_c=word2vec[a],word2vec[b],word2vec[c]
    # 注意余弦相似度越大，越相似
    # 1，-1则意思相反
    maxi=-2
    best_word=None
    words = word2vec.keys()
    for word in words:
        if word in [a,b,c]:
            continue
        cos_sim=cos_similaryty(vec_a-vec_b,vec_c-word2vec[word])
        if cos_sim>maxi:
            maxi=cos_sim
            best_word=word
    return best_word


# In[18]:


test_words=[('man','woman','boy'),('small','smaller','large'),('italy','italian','china'),('Japan','Tokyo','China')]
for words in test_words:
    print(f'{words[0]}-->{words[1]} like {words[2]} --> {caculate_similar(words[0],words[1],words[2],word2vec)}')


# In[ ]:




