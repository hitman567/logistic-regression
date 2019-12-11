
# coding: utf-8

# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[71]:


text=pd.read_csv("data_logistic.txt")


# In[72]:


text.head()


# In[73]:


x=text.iloc[:,:-1]
y=text.iloc[:,-1]


# In[74]:


def normalize(x):
    mins=np.min(x,axis=0)
    maxs=np.max(x,axis=0)
    rng=maxs-mins
    return 1-((maxs-x)/rng)
x=normalize(x)


# In[75]:


def onelistmaker(n):
    listofones = [1]*n
    return listofones
ones=onelistmaker(x.shape[0])
# print(ones)
x['X0']=x.insert(0,'X0',ones)
x['X0']=ones



# In[76]:


x.head()


# In[77]:


x.shape


# In[87]:


theta0=np.zeros(x.shape[1])
theta1=np.ones(x.shape[1])
theta2=np.empty(x.shape[1])
theta2.fill(2)
theta=theta0

learning_rate=0.4
epsilon=0.0001


# In[88]:


def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
Lambda=0
costfn=[]
def gradient_des(x,y,theta,learning_rate,epsilon):
    z=np.dot(x,theta.T)
    h=sigmoid(z)
    cst=cost(h,y)
    change_cost=1
    no_of_iterations=1
    
    while(change_cost>epsilon):
        old_cost=cst
        gradient=np.dot(x.T,(h-y))/y.shape[0]
        theta-=learning_rate*gradient
        z=np.dot(x,theta.T)
        h=sigmoid(z)
        cst=cost(h,y)
        change_cost=old_cost-cst
        no_of_iterations+=1
        costfn.append(cst)
    return theta,no_of_iterations
m=y.shape[0]


# In[89]:


def reg_gradient_des(x,y,theta,learning_rate,epsilon):
    z=np.dot(x,theta.T)
    h=sigmoid(z)
    cst=cost(h,y)+Lambda/(2*m)*sum(theta**2)
    change_cost=1
    no_of_iterations=1
    
    while(change_cost>epsilon):
        old_cost=cst
        gradient=np.dot(x.T,(h-y))/y.shape[0]+(Lambda/m)*theta
        theta-=learning_rate*gradient
        z=np.dot(x,theta.T)
        h=sigmoid(z)
        cst=cost(h,y)+Lambda/(2*m)*sum(theta**2)
        change_cost=old_cost-cst
        no_of_iterations+=1
        costfn.append(cst)
    return theta,no_of_iterations


# In[90]:


# theta,no_of_iterations=gradient_des(x,y,theta,learning_rate,epsilon)
print("Estimated coefficients:")
print(theta)
print("No. of iterations:")
print(no_of_iterations)


# In[91]:


theta,no_of_iterations=reg_gradient_des(x,y,theta,learning_rate,epsilon)
print("Estimated coefficients:")
print(theta)
print("No. of iterations:")
print(no_of_iterations)


# In[92]:


print(costfn)


# In[93]:


i=1
iterlist=[]
for i in range(no_of_iterations-1):
    iterlist.append(i)
    
print(iterlist)


# In[94]:


plt.plot(iterlist,costfn)
plt.show()


# In[95]:


def pred_values(theta,x):
    pred_prob=sigmoid(np.dot(x,theta.T))
    pred_value=np.where(pred_prob>=0.5,1,0) 
    return np.squeeze(pred_value)
y_pred=pred_values(theta,x)
print("Correctly predicted labels:", np.sum(y == y_pred)) 
print("Incorrectly predicted labels:", np.sum(y != y_pred)) 
print("Accuracy:", 100*np.sum(y == y_pred)/(np.sum(y == y_pred)+np.sum(y != y_pred)))

