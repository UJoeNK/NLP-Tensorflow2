#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.layers import Dense,SimpleRNN,Embedding
import tensorflow as tf
import numpy as np
import random


# # 数据预处理

# 加载文本获取恐龙名字。创建字符表，计算样本与字符表的长度。(大小写不区分）

# In[3]:


data=open('./语料库/dinos.txt').read()
data=data.lower()
char=sorted(set(data))
char_num=len(char)
print(f'样本长度{len(data)},字符数量{char_num}')


# 创建对照列表。char2id表示字符映射到数字。id2char表示数字映射到字符.  
# '\n'表示<EOS>，对应0.

# In[4]:


char2id={i:u+1 for u,i in enumerate(char)}
id2char={u+1:i for u,i in enumerate(char)}
char2id,id2char


# 创建训练集

# In[5]:


with open('./语料库/dinos.txt') as f:
    examples = f.readlines()
examples = [x.lower().strip() for x in examples]
maxlen=max([len(i) for i in examples ])
examples[0]


# 将训练集的字符变为数字编码

# In[6]:


X,Y=[],[]
for index in range(len(examples)):
    x =[char2id[ch] for ch in examples[index]]
    y =x[1:]+[char2id["\n"]]
    X.append(x)
    Y.append(y)
X[0],Y[0]


# 将输出padding为同一长度

# In[7]:


X=np.array(X)
Y=np.array(Y)


# In[8]:


padded_X=tf.keras.preprocessing.sequence.pad_sequences(X,maxlen=maxlen,padding='post',value=0)


# In[9]:


padded_Y=tf.keras.preprocessing.sequence.pad_sequences(Y,maxlen=maxlen,padding='post',value=0)


# In[10]:


print(padded_X.shape,padded_Y.shape)


# 将训练集随机打乱

# In[11]:


np.random.seed(3)
np.random.shuffle(X)
np.random.seed(3)
np.random.shuffle(Y)


# In[12]:


X[3],Y[3]


# In[13]:


print(type(padded_X[0]))


# In[14]:


train_db = tf.data.Dataset.from_tensor_slices((padded_X, padded_Y))
train_db=train_db.batch(32,drop_remainder=True)


# In[15]:


train_iter = iter(train_db)
# next() 返回迭代器的下一个项目
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)
print(sample[0][0],sample[1][0])


# ******************

# ----------------------------------------------------------------------

# # 创建模型 

# 创建模型，注意embdding层是vocab_size+1，应为加入了padding 0.
# 最后的softmax 也是vocab_size+1
# 注意是return_sequences=True,应为每一个时刻我们都要产生输出
# ![](pic/dis1.png)

# In[16]:


class My_model(tf.keras.Model):
    def __init__(self,vocab_size,rnn_units):
        super(My_model,self).__init__()
        self.embedding=Embedding(vocab_size+1,5,name='emb')
        self.rnn=SimpleRNN(rnn_units,return_sequences=True,name='rnn')
#         self.d1=Dense(64,activation='relu',name='d1')
        self.d2=Dense(vocab_size+1,activation='softmax',name='d2')
    
    def call(self,x):
        x=self.embedding(x)
        x=self.rnn(x)
#         x=self.d1(x)
        x=self.d2(x)
        return x
        


# In[17]:


model=My_model(char_num,16)


# ## 取样
# ![](pic/dis3.png)

# 我们在一个时刻会得到一个预测，然后我们需要将这个预测结果作为下一个时间点的输入，然后进行下一次预测。
# 我们输出的yt是softmax之后的结果，代表我们预测下一个单词的概率，然后我们需要依照概率进行抽样（注意不像往常依照取argmax，因为这样我们很有可能产生死循环）。

# In[18]:


import random

def sample(model):
    seed=0
    name=[]
    for i in range(5):
        a=[random.randint(1,27)]
        b=tf.expand_dims(a,0)
        ans=[id2char[a[0]].upper()]
        for i in range(20):
            pred=model(b)
            pred=tf.squeeze(pred)
            pred=np.array(pred)
            
            # for grading purposes
            np.random.seed(i+seed) 
        
            idx = np.random.choice(list(range(28)), p=pred.ravel())
            if idx==0 or idx==1:
                break
            next_word=id2char[idx]
            ans.append(next_word)
            a=[char2id[next_word]]
            b=tf.expand_dims(a,0)
            seed+=1
        
        ans=''.join(ans)
        name.append(ans)
    for n in name:
        if n is not None:
            print(n)


# ## 定义优化器和损失
# 我们要将padding 0 位置上产生的损失mask掉

# In[19]:


loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
optimizer=tf.keras.optimizers.Adam(1e-3)
def loss_function(y_true,y_pred):
    # 我们将0mask掉，不计算0的损失
    mask=tf.math.logical_not(tf.math.equal(y_true,0))
    loss=loss_object(y_true,y_pred)
    mask=tf.cast(mask,dtype=loss.dtype)
    loss*=mask
    return tf.reduce_mean(loss)


# ## 进行训练
# 训练的时候，我们不像预测进行采样，因为训练的时候，我们预测的结果很有可能是错的，然后我们传入错误的结果进行预测，那么产生的下一个结果就更加糟糕了。
# 
# 所以我们使用教师强制（teaching force）的方式。将下一个正确的答案输入到模型，然后进行下一次的预测。
# 此外，我们需要对模型进行梯度裁剪，避免梯度爆炸。
# ![](pic/dis2.png)

# In[21]:


@tf.function
def train_step(inp,targ):
    loss=0
    
    with tf.GradientTape() as tape:
        
        
        # 教师强制-将目标词作为下一个输入
        model_input=inp[:,0]
        model_input=tf.expand_dims(model_input,1)
        for t in range(1,targ.shape[1]):
            # 将编码器输出传到解码器
            predictions=model(model_input)
            
            loss+=loss_function(targ[:,t],predictions)
            
            # 使用教师强制
            model_input=tf.expand_dims(targ[:,t],1)
        
        batch_loss=(loss/int(targ.shape[1]))
        
        variables=model.variables

        # 对每一个变量计算梯度
        gradients=tape.gradient(loss,variables)
        
#         print(type(gradients))
#         print(gradients[0])
#         print(gradients[1])
        
        gradients, _ = tf.clip_by_global_norm(gradients,3)
#         gradients = [tf.clip_by_value(gards, -3, 3) for gards in gradients if gards is not None]

        # Apply gradients to variables
        optimizer.apply_gradients(zip(gradients,variables))
        return batch_loss
            


# 

# In[22]:


EPOCHS=15
steps_per_epoch=len(X)//32
for epoch in range(EPOCHS):
    
    total_loss=0
    
    for (batch,(inp,targ)) in enumerate(train_db.take(steps_per_epoch)):
        batch_loss=train_step(inp,targ)
        total_loss+=batch_loss
     
        # 每 2 个周期（epoch），保存（检查点）一次模型
#     if (epoch + 1) % 2 == 0:
#         checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss ))
    print()
    # 在每一次训练周期进行输出，可以查看一开始生成的名字乱七八糟，后来的名字逐渐有规律了
    sample(model)
    print()


# In[ ]:





# In[ ]:




