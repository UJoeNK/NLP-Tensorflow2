#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


def plot_graphs(history,string):
    plt.plot(history.history[string],'b',label='Training '+string,)
    plt.plot(history.history['val_'+string],'y',label='Val '+string)
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend()
    # plt.show()


# ### 设置输入

# as_supervised: bool, if True, the returned tf.data.Dataset will have a 2-tuple structure (input, label) according to builder.info.supervised_keys. If False, the default, the returned tf.data.Dataset will have a dictionary with all the features.
# 
# with_info: bool, if True, tfds.load will return the tuple (tf.data.Dataset, tfds.core.DatasetInfo) containing the info associated with the builder.
# 
# ds_info: tfds.core.DatasetInfo, if with_info is True, then tfds.load will return a tuple (ds, ds_info) containing dataset information (version, features, splits, num_examples,...). Note that the ds_info object documents the entire dataset, regardless of the split requested. Split-specific information is available in ds_info.splits.

# In[3]:


dataset,info=tfds.load('imdb_reviews/subwords8k',with_info=True,
                      as_supervised=True)
train_dataset,test_dataset=dataset['train'],dataset['test']


# In[4]:


dataset.items()


# In[5]:


# with_info=True 返回了dataset_info。
# info的features['text']中包含了编码器
encoder=info.features['text'].encoder
print(f'Vocabulary size:{encoder.vocab_size}')


# 此文本编码器将可逆地对任何字符串进行编码，必要时返回字节编码。
# 可能是 因为词汇表只有8185.可能有些词语没有收录，就只能对字节编码了。

# In[6]:


sample_string="Hello tensorflow! one day I will keep you safe, keep you sound. I promise you!"
encode_string=encoder.encode(sample_string)
print(f'Encode string is: {encode_string}')
origin_string=encoder.decode(encode_string)
print(f'origin string is:{origin_string}')


# In[7]:


origin_string == sample_string


# In[8]:


for index in encode_string:
    print(f'{index} ---> {encoder.decode([index])}')


# 尝试获取所有词汇，但是因为输入的评论不规整。
# 所得的词汇表不是很准

# In[9]:


vocabulary_set=set()
count=0
# sentence是一个truple，0是语句的tensor，1是label的tensor
for sentence in dataset['train']:
    count+=1
    words=encoder.decode(sentence[0].numpy()).split()
    if count%10000==0:
        print(words)
    vocabulary_set.update(words)
len(vocabulary_set),count


# ## 准备训练集

# In[18]:


BUFFER_SIZE=10000
BATCH_SIZE=64


# In[11]:


# train_dataset.output_shapes


# In[12]:


padded_shapes=(
#         tf.TensorShape([6]),
#         tf.TensorShape([])
    [None,],[]
        )


# In[13]:


# train_dataset=train_dataset.shuffle(BUFFER_SIZE)
# train_dataset=train_dataset.padded_batch(BATCH_SIZE,padded_shapes=train_dataset.output_shapes)
# test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

# 这样写更好
train_dataset=train_dataset.shuffle(BUFFER_SIZE)
train_dataset=train_dataset.padded_batch(BATCH_SIZE,padded_shapes=([-1,],[]))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)


# In[ ]:





# ## 创建模型

# In[14]:


model=tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size,64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()


# In[15]:


model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# In[16]:


checkpoint_save_path = "./checkpoint/imdb_LSTM.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=2)


# 本地训练太慢，已经在colab中训练完毕，上方已经加载完成，不需要再训练了

# In[ ]:


histrory=model.fit(train_dataset,epochs=10,validation_data=test_dataset,callbacks=[cp_callback])


# In[17]:


test_loss,test_accuracy=model.evaluate(test_dataset)
print("Test loss:{}, Test accuracy:{}".format(test_loss,test_accuracy))


# 下面对我们自己的评论进行预测  
# 因为我们的训练集都进行了pad，如果我们自己的评论没有进行pad，在预测时会有较大的影响

# In[19]:


def pad_to_size(vec,size):
    zeros=[0]*(size-len(vec))
    vec.extend(zeros)
    return vec


# In[20]:


def sample_predict(sentence,ispad):
    encoded_sample_pred_text=encoder.encode(sentence)
    if ispad:
        encoded_sample_pred_text=pad_to_size(encoded_sample_pred_text,64)
    encoded_sample_pred_text=tf.cast(encoded_sample_pred_text,tf.float32)
    predictions=model.predict(tf.expand_dims(encoded_sample_pred_text,0))
    return predictions


# In[24]:


sample_pred_texts =['this movie is bad. but the actor is very handsome and I like him. but I will not recommend this movie.',
                   'The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.',
                    "actually, I am the actor's fans. But his performance in the movie break my heart.",
                    'The characters is not famous, but their performances make the movie reach a very high level! ',
                    'The movie is very ironic.This film criticizes the social phenomena without conscience'
                    ]
for sample_pred_text in sample_pred_texts:
    prediction_without_padding=sample_predict(sample_pred_text,ispad=False)
    prediction_with_padding=sample_predict(sample_pred_text,ispad=True)
    print('comment is:',sample_pred_text)
    print('here is the point:')
    print('padding result is :',prediction_with_padding)
    print('no padding result is :',prediction_without_padding)
    print('******')


# In[ ]:





# In[ ]:





# In[ ]:




