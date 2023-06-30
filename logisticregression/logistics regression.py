#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[2]:


dataset = pd.read_csv('binary_log.csv')


# In[3]:


dataset


# In[4]:


dataset['Admitted'] = dataset['Admitted'].map({'Yes':1,'No':0})


# In[5]:


dataset


# In[6]:


X = dataset.iloc[:,0].values


# In[7]:


X = X.reshape(168,1)


# In[8]:


X.shape


# In[9]:


X


# In[10]:


y = dataset.iloc[:,1].values


# In[11]:


y


# In[12]:


plt.scatter(X,y)
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.show()


# In[13]:


X = dataset.iloc[:,0].values


# In[14]:


X.shape


# In[15]:


X = X.reshape(168,1)


# In[16]:


X


# In[17]:


y = dataset.iloc[:,1].values


# In[18]:


y


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[21]:


X_train


# In[22]:


X_train.shape


# In[23]:


X_test


# In[24]:


X_test.shape


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


model = LogisticRegression()


# In[27]:


model.fit(X_train,y_train)


# In[28]:


#evaluatio0n


# In[29]:


y_pred = model.predict(X_test)


# In[30]:


#prediction
y_pred


# In[31]:


# actual values
y_test


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


print(confusion_matrix(y_test,y_pred))


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


print(classification_report(y_test,y_pred))


# In[ ]:




