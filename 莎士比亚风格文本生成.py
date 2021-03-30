#!/usr/bin/env python
# coding: utf-8

# 说明：此生成模型是基于字符的。训练一个模型预测该序列的下一个字符，重复调用该模型，从而生成更长的文本序列。
# 
# 虽然有些句子符合语法规则，但是大多数句子没有意义。这个模型尚未学习到单词的含义，
# 
# 此模型是基于字符的。训练开始时，模型不知道如何拼写一个英文单词，甚至不知道单词是文本的一个单位。
# 输出文本的结构类似于剧本 -- 文本块通常以讲话者的名字开始；而且与数据集类似，讲话者的名字采用全大写字母。
# 如下文所示，此模型由小批次 （batch） 文本训练而成（每批 100 个字符）。即便如此，此模型仍然能生成更长的文本序列，并且结构连贯。

# In[1]:


import tensorflow as tf
import numpy as np
import os
import time


# In[2]:


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# ### 读取数据

# In[3]:


# 读取数据并解码
text=open(path_to_file,'rb').read().decode(encoding='utf-8')
print("length of text:{} characters.".format(len(text)))


# In[4]:


print(text[:349])


# In[5]:


# 查看文章中的字符
vocab=sorted(set(text))
print('unique characters:{}'.format(len(vocab)))


# ## 处理文本

# ### 向量化文本

# 我们首先要将字符串映射到数字表示，创建两个字典
# 一个用于查找字符映射到数字，一个用于将数字映射到字符

# In[6]:


vocab[:15]


# In[7]:


# 字符到数字
char2idx={u:i for (i,u) in enumerate(vocab)}
# 数字映射到字符
idx2char=np.array(vocab)
# 将文本转换为数字
text_as_int=np.array([char2idx[c] for c in text])
len(text_as_int)


# In[8]:


# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)     # 打包为元组的列表
# [(1, 4), (2, 5), (3, 6)]
zip(a,c)              # 元素个数与最短的列表一致
# [(1, 4), (2, 5), (3, 6)]
zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# [(1, 2, 3), (4, 5, 6)]


# In[9]:


# 查看前20个字符映射到数字
print('{')
for char,_ in zip(char2idx,range(20)):
    #repr() 函数将对象转化为供解释器读取的形式。
    print("{:4s}: {:3d},".format(repr(char),char2idx[char]))
print('...}')


# In[10]:


# 显示文本前13个字符映射到数字
print(f'{text[:13]} --- mapping to num --->{[char2idx[c] for c in text[:13]]}')


# 文件读取完成之后，我们需要明白我们的任务。
# 我们读取一个字符，然后预测下一个最有可能的字符。
# 我们需要训练一个模型用于预测每一个时间步最有可能的输出

# ## 创建训练样本

# 将文本切分为样本序列，每个输入序列包含文本的seq_lebgth个字符
# 对于每一个输入序列，其目标序列包含对应长度相同的文本，但是向右移一个字符。
# 
# 将文本拆分为长度为seq_length+1的文本块。假设seq_length=4,文本为‘hello’。那么输入序列为‘Hell',目标序列为'ello'。
# 首先使用tf.data.Dataset.from_tensor_slices将文本向量转为字符索引流

# In[ ]:





# In[11]:


# 设定句子长度
seq_length=100
example_per_epoch=len(text)//seq_length

# 创建训练样本/目标
char_dataset=tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(i)
    print(idx2char[i.numpy()])


# In[12]:


# 我们可以使用batch方法，将单个字符转换为所需要的长度序列
# 注意不是pad_batch
# 因为dataset中每一个字符就是一个样本，dataset是一个总长为1115394
# 这就表示我们有1115393个输入字符
# 有1115393个目标字符
# 我们使用batch方法，将每一条序列长度变为101，这就是我们一条原始的文本
# 之后我们再将其变为输入文本与输出文本

# 注意每条序列长度是seq_length+1
# 输入长度是seq_length,目标长度是seq_length
sequences=char_dataset.batch(seq_length+1,drop_remainder=True)

for item in sequences.take(2):
    print(item)
    print(repr(''.join(idx2char[item.numpy()])))


# 接下来要创建训练样本与输出目标样本
# 对于每一个序列，我们使用map方法先复制，在顺移一位，从而创建输入与目标
# map函数可以将一个函数应用到每一个batch上

# In[13]:


def split_input_target(chunk):
    input_text=chunk[:-1]
    target_text=chunk[1:]
    return input_text,target_text
dataset=sequences.map(split_input_target)


# In[14]:


# dataset中包含了输入与目标文本
dataset


# In[15]:


for input_example,target_example in dataset.take(1):
    print('input is: {}'.format(repr(''.join(idx2char[input_example.numpy()]))))
    print('target is: {}'.format(repr(''.join(idx2char[target_example.numpy()]))))


# 在每一个时间步，会从input中接受一个字符，然后预测目标是target中对应索引的字符。如输入'F'，尝试预测‘i’。在下一个时间步，接受‘i’，尝试预测‘r’。使用RNN，不仅会考虑当前时间步的输入，也会考虑之前时间步的输入。

# In[16]:


for i,(input_idx,target_idx) in enumerate((zip(input_example[:5],target_example[:5]))):
    print(f'timestep {i}:')
    print(f'  the input is :{idx2char[input_idx]}')
    print(f'  the target is:{idx2char[target_idx]}')


# ### 创建训练批次

# In[17]:


BATCH_SIZE=64

# 设置缓冲区大小，以重新排列数据集
# （TF 数据被设计为可以处理可能是无限的序列，
# 所以它不会试图在内存中重新排列整个序列。相反，
# 它维持一个缓冲区，在缓冲区重新排列元素。） 
BUFFER_SIZE=10000

dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
# 现在dataset每条输入样本长为100，一批样本为64条
# dataset=dataset.batch(BATCH_SIZE,drop_remainder=True)


dataset


# In[18]:


text_batch=[]
for i in dataset.take(3):
    text_batch.append(i[0].numpy())
    text_batch.append(i[1].numpy())


# In[19]:


for i in range(1):
    print(''.join(idx2char[text_batch[0][0]]))
    print('---')
    print(''.join(idx2char[text_batch[0][1]]))


# ## 创建模型

# In[20]:


vocab_size=len(vocab)
embedding_dim=256
rnn_units=1024
batch_size=BATCH_SIZE


# 关于为什么要在embedding层加入batch_input_shape  
# ValueError: If a RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: 
# - If using a Sequential model, specify the batch size by passing a `batch_input_shape` argument to your first layer.
# - If using the functional API, specify the batch size by passing a `batch_shape` argument to your Input layer.

# 下面是一些关于GRU的参数说明

# - stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
# 即会将上一个批次的最后一个状态作为下一个批次的初始状态

# - return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False  
# 是返回输出序列中的最后一个输出，还是返回完整序列
# ==true输出全部，false输出最后一个
# 控制的是hidden_state
# 
# 注意区别return_state:控制的是cell_state.
# return_state=true输出，false不输出

# 此处有一个疑问，使用了stateful=true  
# 但是数据集连续的batch之间同一个index i 处的数据我觉得并没有连续

# In[21]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size,activation='softmax')
  ])
    return model


# In[37]:


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# ![image.png](./pic/text_generate1.png)

# 关于logits的理解
# 也可以这么理解：logits与 softmax都属于在输出层的内容  
# logits = tf.matmul(X, W) + bias  
# 再对logits做归一化处理，就用到了softma:  
# Y_pred = tf.nn.softmax(logits,name='Y_pred')
# 
# 

# ![](pic/text_generation2.png)

# ## 测试模型

# In[23]:


# 检查模型输出形状
for input_example_batch,target_example_batch in dataset.take(1):
    example_batch_predictions=model(input_example_batch)
    print(example_batch_predictions.shape) #(batch_size,seq_length,vocab_size)b


# In[24]:


print(sum(example_batch_predictions[0][2]))


# 上面的例子虽然长度为100，但是我们的模型可以处理任何长度，因为在embedding层的batch_input_shape设置为了[batch_size,None],在model.summary()中查看

# In[25]:


model.summary()


# 为了进行预测，我们还要对模型的输出进行抽样（sample），从而获得输出的字符。
# 这个分布是更具字符集的逻辑回归进行定义的  
# 需要注意，进行sample是必要的，否则的话取分布的最大值的索引，这样模型就很有可能卡在循环中。
# 

# tips：
# 使用tf.random.categorical从一个分类分布中抽取样本
# ```
# tf.random.categorical(
#     logits,
#     num_samples,
#     dtype=None,
#     seed=None,
#     name=None
# )
# ```
# - logits: 形状为 [batch_size, num_classes]的张量. 每个切片 [i, :] 代表对于所有类的未正规化的log概率。
# - num_samples: 0维，从每一行切片中抽取的独立样本的数量。  
# 返回是样本的索引
# 

# In[26]:


sample_indeces=tf.random.categorical(example_batch_predictions[0],num_samples=1)
sample_indeces=tf.squeeze(sample_indeces,axis=-1).numpy() # tf.squeeze删除一个维度
# 输出sample_indeces,即为我们依据分布进行抽样得到的下一个预测字符的索引
sample_indeces


# In[27]:


# 将索引转换为字符，查看未训练之前所得到的输出
print('input data:{}'.format(repr(''.join(idx2char[input_example_batch[0]]))))
print('......')
print('prediction without training:{}'.format(repr(''.join(idx2char[sample_indeces]))))


# 可以看到，为训练之前，我们的模型输出是杂乱无章的，不知道单词的组成，文本的格式等等，跟不用说语法与语义

# ## 训练模型

# 我们的模型可以看作一个多分类问题，最后预测下一个字符的类别  
# 
# 
# #### 一个疑问，为什么最后一层不使用softmax激活函数，这样最后就不需要对输出进行逻辑回归将其转化为概率进行抽样

# In[28]:


# 由于模型返回的是逻辑回归，所以我们需要设定参数from_logits
def loss(labels,logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
exapmle_batch_loss=loss(target_example_batch,example_batch_predictions)
print(f'example mean loss:{exapmle_batch_loss.numpy().mean()}')


# In[29]:


model.compile(optimizer='adam',loss=loss)


# In[30]:


checkpoint_save_path = "./text_generation_checkpoint/text_generation.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=2)


# In[ ]:


EPOCHS=5


# In[ ]:


# 已经在colab上训练了30个epoch
history = model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])


# In[ ]:





# ## 生成文本

# ### 回复检查点

# 我们将批大小设置为1，由于之前训练的时候设定的批大小为64，我们要使用不同的batch_size。我们需要重新建立模型，从checkpoint恢复权重即可

# In[23]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

checkpoint_save_path = "./text_generation_checkpoint/text_generation.ckpt"

# 加载训练好的模型，本地训练太慢了，在colab中训练完毕了
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    print(checkpoint_save_path)
    model.load_weights(checkpoint_save_path)

# choose to manually build your model by calling `build(batch_input_shape)`:
model.build(tf.TensorShape([1, None]))


# In[24]:


model.summary()


# ### 循环预测

# - 首先我们需要设置起始字符串，初始化RNN的状态，然后设置生成字符的个数
# - 然后利用起始字符串和RNN的状态，获取下一个字符的预测分布
# - 然后使用分类分布计算预测字符串的索引，然后将这个字符串当作模型下一个时间步的输入
# - 模型返回的 RNN 状态被输送回模型。现在，模型有更多上下文可以学习，而非只有一个字符。在预测出下一个字符后，更改过的 RNN 状态被再次输送回模型。模型就是这样，通过不断从前面预测的字符获得更多上下文，进行学习。  
# 
# ![](pic/text_generation3.png)

# 之后我们会发现，我们的模型已经能够写出正确格式的莎士比亚风格诗句，知道什么时候大写，什么时候空格，什么时候分段。当然，对于语法和语义还没有学习到什么

# In[49]:


def generate_text(model,start_string):
    num_generate=1000 #生成字符数
    
    # 将起始字符转换为数字
    input_eval=[char2idx[s] for s in start_string]
    # 增加一个维度，并且可将输入变为张量
    input_eval=tf.expand_dims(input_eval,0)
    
    # 存储结果
    text_generated=[]
    
    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过试验以找到最好的设定
    
    # 更高的温度得到的是熵更大的采样分布，会生成更加出人意料、更加无结构的生成数据，
    # 而更低的温度对应更小的随机性，以及更加可预测的生成数据。
    temperature = 1.0
    
    model.reset_states()
    
    for i in range(num_generate):
        
        # 此时shape是 [batch_size=1,seq_length,voacb_size]
        predictions=model(input_eval)
        

        # 此时shape是 [seq_length,voacb_size]
        predictions=tf.squeeze(predictions,0)
        
#         pred=tf.keras.activations.softmax(predictions).numpy()
#         print(pred.shape)
        
        
        # 依据分布进行抽样
        predictions=predictions/temperature
        # tf.random.categorical返回的是一个二维的tensor
        # shape=(batch_size,num_samples)
        # [-1,0]即取返回值的最后一个batch_size的第一个元素
        # 因为我们输入可能是多个字符，如‘ROME’，输出维度就是（4,vocab_size=65)
        # 所以我们用[-1,0]来获得“ROME’中最后一个‘E’的下一个抽样产生的输出（sample）
        prediction_index=tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

        
#         pred=np.array(pred)[-1,:]
#         print(pred.shape)
        # p代表每个元素选取的概率
#         prediction_index = np.random.choice(list(range(65)), p=pred.ravel())
        
        # 将上一个预测的字符和之前的状态传入模型，作为下一个输入
        input_eval=tf.expand_dims([prediction_index],0)
        text_generated.append(idx2char[prediction_index])
        
    return start_string +''.join(text_generated)


# In[50]:


for i in tf.range(10):
    samples = tf.random.categorical([[1.0,1.0,1.0,1.0,1.0]], 1)
    print(samples)


# In[51]:


print(generate_text(model, start_string=u"ROMEO: "))
# 很奇怪，权重都是保存在谷歌云盘上，下载下来的，在本地结果很糟糕
# 在colab上加载相同的模型权重
# 效果如下：
'''
ROMEO: I advance fiture each other,
How many haughty love, your own suspicion from so rounder he divide,
As if I had some all fell.

Fullow:
Bleased the soldiers, Cleome,
And thou hadst beat me back to Man
In an outward stars that sle with thee?
Why should she noble endary?

DUKE OF YORK:
'Twas something I have you aud in France,
And rear ourselves: 'tis he that lives in the substance where
They are buts for a schollow.

CAPULET:
God and for all his own good will doth lack some general.

Gire descings beasts do go.

LADY GREY:
My lords, so amel, or ho! You are plack'd,
And nother ready straight
And ragers else to make in piece of my mind.

WARWICK:
Ay for my middless sin with arms:
Be you, covert:
We cannot blow our needs, even whether I wear your highness
Will up my master read it in his high;
To-morrow or perpetual speech, have you know the drowsy overworn:
When I would be the rest receive an offer;
Why, why, your fearful souls thy head,
And errs as swiftly, sir;
Hortensio after largers, fr

'''


# In[ ]:




