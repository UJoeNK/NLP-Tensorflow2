#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
import tensorflow as tf


# In[2]:


# 用于生成数据，生成一些格式的日期 

fake = Faker()
fake.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    
    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}
 
    return dataset, human, machine, inv_machine


# In[3]:


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)


# In[4]:


dataset[:10]


# In[5]:


# human_vocab,machine_vocab  对应每个字符的数字编码
# inv_machine_vocab 数字对应的字符
human_vocab,machine_vocab,inv_machine_vocab


# In[6]:


# 对输入的日期字符串编码为数字
def string_to_int(string,maxlen,vocab):
    string=string.lower()
    string=string.replace(',','')
    
    # 规定最大长度
    if len(string)>maxlen:
        string=string[:maxlen]
    int_code=list(map(lambda x:vocab.get(x,'<unk>'),string))
    
    # 如果长度不够，我们加上pad
    if len(string)<maxlen:
        int_code+=[vocab['<pad>']]*(maxlen-len(string))
    return int_code


# In[7]:


#tf.keras.utils.to_categorical 将类别编码向量转换为二进制矩阵（onehot矩阵）
y = [0, 1, 2, 3] 
tf.keras.utils.to_categorical(y, num_classes=4) 


# In[8]:


# 将数字编码向量的输入转换为onehot向量（这样就不用embedding了）
# 应为总共输入词汇表大小为37，维度很小，用onehot向量即可
def preprocessing_data(data,maxlen,vocab):
    data=np.array([string_to_int(i,maxlen,vocab) for i in data])
    # map遍历data中的每一个值，将其作用于to_categorical（）函数，转换为onehot向量
    data_onehot=np.array(list(map(lambda x:tf.keras.utils.to_categorical(x,num_classes=len(vocab)),data)))
    return data,data_onehot


# In[9]:


Tx=30 # 输入长度最大设置为30
Ty=10 # 1990-01-01 输出长为10，输出的长度是固定的


# In[10]:


X,Y=zip(*dataset)# 解压dataset，每个元组第一个值分配到X，第二个分配到Y
X,Xoh=preprocessing_data(X,Tx,human_vocab)
Y,Yoh=preprocessing_data(Y,Ty,machine_vocab)


# In[11]:


index = 100
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
print(X.shape,Y.shape,Xoh.shape,Yoh.shape)


# In[12]:


from tensorflow.keras.layers import Bidirectional,Concatenate,Dot,Input,LSTM,Multiply
from tensorflow.keras.layers import RepeatVector,Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


# ![pic](pic/attention_date1.png)

# Attention说明：
# - 底层是双向LSTM，然后向Attention层传入所有时刻的隐状态  ($[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$) 
# - 在attention层，我们传入上层LSTM的前一时刻隐状态 s，将其通过repeator复制Tx份，然后与隐状态($[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$)  拼接（concat），之后通过全连接层得到et，之后通过softmax层计算出当前时刻对每一个输入单词的attention weights ($[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$)。
# - 然后得到attention向量:  $context^{<t>} = \sum_{t' = 0}^{T_x} \alpha^{<t,t'>}a^{<t'>}$。然后作为输入传入上层LSTM网络。
# 
# __注意：在上层网络中，没有将yt-1作为输入。因为日期翻译中，上一个单词与下一个单词没有什么关系。但在文本翻译，文本生成等任务中就不一样了__

# 然后我们带着公式过一遍：
# 
# ![](pic/attention_date2.png)

# In[13]:


# 定义一些组件，Attention的时候用到
repeator=RepeatVector(Tx)
concatenator=Concatenate(axis=-1)
densor1=Dense(10,activation='tanh',name='Dense1')
densot2=Dense(1,activation='relu',name='Dense2')
# activator=Activation(softmax)
dotor=Dot(axes=1)


# In[14]:


# 对一个时间步进行attention计算
def one_step_attention(a,s_prev):
    """
    a:底层双向LSTM的隐状态 维度：（m，Tx，2*na）
    s_prev：上层LSTM的前一时刻隐状态  维度：(m,ns)
    """
    # 复制s
    s_prev=repeator(s_prev)
    # 与a相拼接
    concat=concatenator([a,s_prev])
    # 计算得到et
    e=densor1(concat)
    
    e=densot2(e)
    # 计算attention weights
    alphas=tf.nn.softmax(e,axis=1)
    # 得到context vector
    context=dotor([alphas,a])
    
    return context


# In[15]:


na=32
ns=64
# return state=true 会返回最后时刻的细胞状态c
post_LSTM_cell=LSTM(ns,return_state=True)
output_layer=Dense(len(machine_vocab),activation='softmax')


# In[16]:


# 对于LSTM使用了解不够
# 可以运行一下注释
# 然后，删除return_sequences=True，在运行

# inputs = tf.random.normal([32, 10, 8]) 
# lstm = tf.keras.layers.LSTM(4) 
# output = lstm(inputs) 
# print(output.shape) 

# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True) 
# whole_seq_output, final_memory_state, final_carry_state = lstm(inputs) 
# print(whole_seq_output.shape) 

# print(final_memory_state.shape) 

# print(final_carry_state.shape) 

# print(whole_seq_output) 

# print(final_memory_state) 
# 注意final_memory_state会等于whole_seq_output的最后一个值


# 这里我们选择函数式模型（model)，所以不需要提前实例化，先将网络结构实现:
# 

# In[17]:


def model(Tx,Ty,na,ns,human_vocab_size,machine_vocab_size):
    """
    Tx:输入的序列长度
    Ty:输出的序列长度
    na:双向LSTM的隐状态维度
    ns:上层LSTM的隐状态维度
    human_vocab_size:输入的词汇表大小
    machine_vocab_size:输出的词汇表大小
    """
    # 定义输入：X 日期的onehot向量
    # s0，c0（LSTM的状态值）
    X=Input(shape=(Tx,human_vocab_size),name='X')
    s0=Input(shape=(ns,),name='s0')
    c0=Input(shape=(ns,),name='c0')
    
    s=s0
    c=c0
    # 存放输出
    outputs=[]
    # 底层双向LSTM
    a=Bidirectional(LSTM(na,return_sequences=True,name='bidirectional'),merge_mode='concat')(X)
    
    # 10个时间步，产生输出长度为10 如：1990-01-20
    for t in range(Ty):
        
        context=one_step_attention(a,s)
        # 使用上一时刻的状态初始化LSTM
        s,_,c=post_LSTM_cell(context,initial_state=[s,c])
        
        # output_layer=Dense(len(machine_vocab),activation='softmax')
        out=output_layer(s)
        
        outputs.append(out)
        
#     print(len(outputs),outputs[0].shape)
#               10           (None, 11)
    # 定义模型具有一个三个输入，一个输出
    model=Model(inputs=(X,s0,c0),outputs=outputs)
    
    return model


# In[18]:


# 实例化模型
model=model(Tx,Ty,na,ns,len(human_vocab),len(machine_vocab))


# In[19]:


model.summary()


# In[20]:


opt=Adam(lr=0.005, beta_1=0.9, beta_2=0.995, epsilon=None, decay=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


# In[21]:


#初始化LSTM状态，全0
s0=np.zeros((m,ns))
c0=np.zeros((m,ns))
# 注意到模型的outputs是一个包含11个元素的列表，每一个元素维度是（m，Ty）。
# 即outputs的维度是（10，10000，11）
# 所以我们要将Yoh（10000，10，11）的轴进行交换，维度变为（10，10000，11）
outputs=list(Yoh.swapaxes(0,1))
print(Yoh.shape)
print(len(outputs),len(outputs[0]),len(outputs[0][0]))


# In[22]:


model.fit([Xoh,s0,c0],outputs,epochs=6,batch_size=64)


# In[24]:


# 进行预测

def predict(example):
    # 因为预测每次输入一个用例，所以初始化LSTM细胞和隐状态维度为（1，ns）
    s0 = np.zeros((1, ns))
    c0 = np.zeros((1, ns))
    # 将输入日期变为onehot向量
    source=string_to_int(example,Tx,human_vocab)
    source=np.array(list(map(lambda x:tf.keras.utils.to_categorical(x,num_classes=len(human_vocab)),source)))
    source=tf.expand_dims(source,0)
#     print(source.shape)
    
    prediction=model.predict([source,s0,c0])
#     print(len(prediction))
#     print(prediction[0].shape)
    # 获得输出，使用argmax获得概率最大的预测值作为输出
    prediction=np.argmax(prediction,axis=-1)
#     print(prediction)
    # 将预测的数字转换为字符
    output=[inv_machine_vocab[int(i)] for i in prediction]
    print('source:',example)
    print('output:',''.join(output))
    print()


# In[46]:


examples=['3/may/1979','18.4.2009','04 22 2004','6th of August 2016','Tue 10 Jul 2020','March 4 2009','12/23/2001','monday march 7 2013']
for example in examples:
    predict(example)


# In[48]:


for example in dataset[100:120]:
    predict(example[0])


# In[ ]:




