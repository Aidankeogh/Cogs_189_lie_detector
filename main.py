
# coding: utf-8

# In[1]:


from file_loader import getAllAnswers

sequence_dict, category_dict = getAllAnswers() 


# In[2]:


X = []
Y = []
for key in sequence_dict.keys():
    for seq, cat in zip(sequence_dict[key],category_dict[key]):
        X.append(seq)
        Y.append(cat)


# In[3]:


# Z score data
import numpy
n = [[] for i in range(8)]

for question in X:
    for timestep in question:
        for i in range(len(timestep)):
            n[i].append(timestep[i])
            
arr = numpy.array(n)
means = numpy.mean(arr, axis=1)
stdevs = numpy.std(arr, axis=1)

for question in X:
    for timestep in question:
        for i in range(len(timestep)):
            timestep[i] = (timestep[i] - means[i])/stdevs[i]


# In[ ]:


import torch.nn as nn
from torch.autograd import Variable
from Seq_GRU import GRU
import random
import torch
random.seed(10)
torch.manual_seed(10)

rnn = GRU(8, 32, 2, learning_rate = 0.0001).cuda()


# In[ ]:


rnn.fit(X,Y,print_every = 1000)

