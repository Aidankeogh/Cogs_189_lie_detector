
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, print_function, division 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import random

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def toTensor(r):
    X, Y = r
    Y = Variable(torch.LongTensor([Y])).cuda()
    X = Variable(torch.FloatTensor([[x] for x in X])).cuda()
    return (X, Y)

def randLoader(R):
    while(True):
        random.shuffle(R)
        for r in R:
            yield (toTensor(r))

def dualLoader(X, Y, v_split):
    R = zip(X, Y)
    random.shuffle(R)
    i = int(len(R) * v_split)
    train = R[i:]
    test = R[:i]
    print (len(train),len(test))
    return randLoader(train), randLoader(test)
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.0005):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size,hidden_size)
        
        self.drp = nn.Dropout(p=0.3)
        
        self.fc = nn.Linear(hidden_size,output_size) 
        self.softmax = nn.LogSoftmax()
        
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(),learning_rate)
        
    def forward(self, input):
        hidden = self.initHidden()
        
        for step in input:
            hidden = self.cell(step, hidden)
            hidden = self.drp(hidden)
        
        output = self.fc(hidden)
        output = self.softmax(output)
        
        return output

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).cuda()
    
    def trainSample(self, category_tensor, seq_tensor, validation = False):
        hidden = self.initHidden().cuda()
        
        self.optimizer.zero_grad()
        output = self(seq_tensor)
        
        loss = self.criterion(output, category_tensor)
        
        values, indices = output.max(1) 
        
        correct = (indices == category_tensor)
        if(not validation):  
            loss.backward()
            self.optimizer.step()

        return output, loss.data[0], correct.data[0]

    def fit(self, X, Y, n_iters = 1000000, print_every = 2500, v_split = 0.2):
        current_loss = 0
        current_correct = 0
        v_loss = 0
        v_corr = 0
        all_losses = []
        v_losses = []
        start = time.time()
        loader, valid_loader = dualLoader(X, Y, v_split)
        print('iter|perc|time|loss_t|loss_v|acc_t|acc_v')
        for iter in range(1, n_iters + 1):
            seq, category = loader.next()  
            output, loss, correct = self.trainSample(category, seq)
            current_loss += loss
            current_correct += correct
            
            vs, vc = valid_loader.next()
            vo, vl, vcorrect = self.trainSample(vc, vs, validation=True)
            #print(vcorrect)
            v_loss += vl
            v_corr += vcorrect

            # Print iter number, loss, name and guess
            if iter % print_every == 0:   
                print('%d \t %.1f%% \t %s \t %.4f \t %.4f \t %.2f \t %.2f ' % (iter, iter / n_iters * 100, timeSince(start), 
                                        current_loss / print_every, v_loss / print_every, 
                                        current_correct / print_every, v_corr / print_every))
                
                all_losses.append(current_loss / print_every)
                v_losses.append(v_loss / print_every)
                v_loss = 0
                v_corr = 0
                current_loss = 0
                current_correct = 0
        return all_losses


# In[ ]:












