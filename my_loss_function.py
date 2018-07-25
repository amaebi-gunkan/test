import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
from chainer import Variable
from chainer.datasets import tuple_dataset
from chainer.cuda import to_cpu
import os

def cross_entropy_for_each(y,t):

    #t_temp = cp.array(to_cpu(t))
    t_temp = cp.array(t)

    t_binaly = cp.zeros(t_temp[0].size).reshape(t_temp[0].shape)
    t_binaly = t_binaly[None, ...]
    loss = cp.zeros(y.shape[1])
    for j in range(y.shape[0]):
        y2 = y[j]
        y2 = y2[None, ...]
        t2 = t_temp[j]
        t2 = t2[None, ...]
    
        ysoft = F.softmax(y2)
        ysoft = cp.array(ysoft.data)
        y2 = cp.array(y2.data)

       # ysoft = cp.array(to_cpu(ysoft.data))
        #y2 = cp.array(to_cpu(y2.data))
        y_2ch = cp.zeros(y2.shape[2]*y2.shape[3]*y2.shape[4]).reshape(1,1,y2.shape[2],y2.shape[3],y2.shape[4])

        for i in range(len(y[0])):


            y_2ch[0,0] = ysoft[0,i] 
            y_2ch = -cp.log(y_2ch)  
            y_2ch[0,0] = cp.where(y_2ch[0,0] == np.inf,(cp.max(y2,axis=1)-y2[0,i]),y_2ch[0,0])
            t_binaly = cp.where(t2 == i,1,0)

            loss[i] += cp.mean(y_2ch[0,0] * (t_binaly))

    loss = loss/y.shape[0]

    return loss


def DiceLossFunction(y,t):
    
    loss = 0.0
    div =  cp.float32(y.shape[0] * y.shape[1])
    y = F.softmax(y,axis=1)

    eps = 0.0001

    for i in range(y.shape[0]):
        soft=y[i]
        tb = cp.array(t[i].flatten())
        for j in range(y.shape[1]):

            V_in = cp.where(tb == j,1,0).astype(cp.float32)
            if (cp.sum(V_in) == 0.0):
                div -=1.0
            t_temp = chainer.Variable(V_in)
            soft_temp = F.flatten(soft[j])

            loss += 2.0*F.sum(soft_temp*t_temp)/(F.sum(soft_temp + t_temp) + eps)
    loss = loss/div

    return -loss   


#def DiceLossFunction(y,t,each_out = False):
    
#    loss = 0.0
#    y = F.softmax(y,axis=1)
#    loss_each=[]
#    for i in range(y.shape[0]):
#        soft=y[i]
#        tb = t[i].flatten()
#        loss_each.append([])
#        for j in range(y.shape[1]):
            
#            t_temp = chainer.Variable(cp.where(tb == j,1,0).astype(cp.float32))

#            soft_temp = F.flatten(soft[i])

#            loss_temp = 2*F.sum(soft_temp*t_temp)/(F.sum(soft_temp + t_temp)+1)
#            loss += loss_temp
#            loss_each_temp = loss_temp
#            #print(F.flatten(loss_each_temp).data)
#            loss_each_temp2 = F.flatten(loss_each_temp).data[0]

#            loss_each[i].append(to_cpu(loss_each_temp2.flatten()[0]))
      
#    loss = loss/y.shape[0]

#    if(each_out is True):
#        return -loss ,loss_each
#    elif(each_out is False):
#        return -loss   

#多分2クラス用
def DiceLossFunction2(y,t):
    
    dice_numerator=0.0
    dice_denominator=0.0
    eps = 0.0001
    div = cp.float32(y.shape[0] * y.shape[1])
    
    y = F.softmax(y,axis=1)

    for i in range(y.shape[0]):#batch-size
        soft = y[i]
        tb = cp.array(t[i].flatten()).astype(cp.float32)
        for j in range(y.shape[1]):#class-size

            V_in = cp.where(tb == j,1,0).astype(cp.float32)
            t_temp = chainer.Variable(V_in)
            soft_temp = F.flatten(soft[j])
            
            dice_numerator += F.sum(soft_temp * t_temp)
            dice_denominator += F.sum(soft_temp + t_temp)
            
    loss = 2.0 * dice_numerator / (dice_denominator+eps)

    return -loss




def GeneralizedDiceLossFunction(y,t,w):
    

    dice_numerator=0.0
    dice_denominator=0.0
    eps = 0.0001
    div = cp.float32(y.shape[0] * y.shape[1])
    
    y = F.softmax(y,axis=1)
    for i in range(y.shape[0]):#batch-size
        soft = y[i]
        tb = cp.array(t[i].flatten()).astype(cp.float32)
        for j in range(y.shape[1]):#class-size
            wb = cp.array(w[i][j].flatten()).astype(cp.float32)
            V_in = cp.where(tb == j,1,0).astype(cp.float32)

            t_temp = chainer.Variable(V_in)
            w_temp = chainer.Variable(wb)
            soft_temp = F.flatten(soft[j])

            dice_numerator += F.sum(w_temp * soft_temp * t_temp)
            dice_denominator += F.sum(w_temp * (soft_temp + t_temp))

    loss =  2.0 * dice_numerator / (dice_denominator+eps)

    return -loss

def DiceLossFunction_weighted(y,t,w):
    
    loss = 0.0
    div =  cp.float32(y.shape[0] * y.shape[1])
    y = F.softmax(y,axis=1)

    eps = 0.0001

    for i in range(y.shape[0]):
        soft=y[i]
        tb = cp.array(t[i].flatten())
        for j in range(y.shape[1]):
            wb = cp.array(w[i][j].flatten()).astype(cp.float32)
            V_in = cp.where(tb == j,1,0).astype(cp.float32)
            if (cp.sum(V_in) == 0.0):
                div -=1.0
            t_temp = chainer.Variable(V_in)
            w_temp = chainer.Variable(wb)
            soft_temp = F.flatten(soft[j])

            loss += 2.0*F.sum(w_temp*soft_temp*t_temp)/(F.sum(w_temp*(soft_temp + t_temp)) + eps)
    loss = loss/div

    return -loss   

#for one hot label
def dice_coefficent(self,predict, ground_truth):
       dice_numerator = 0.0
       dice_denominator = 0.0
       eps = 1e-16

       predict = F.flatten(predict)
       ground_truth = F.flatten(ground_truth.astype(np.float32))

       dice_numerator = F.sum(2*(predict * ground_truth))
       dice_denominator =F.sum(predict+ ground_truth)
       loss = dice_numerator/(dice_denominator+eps)

       return -loss