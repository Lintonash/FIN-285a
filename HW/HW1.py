#!/usr/bin/env python
# coding: utf-8

# In[30]:


list1 = list(range(12,101))
list2 = []
for i in range(len(list1)):
    if i%2 == 0:
        list2.append(list1[i])
sum1 = sum(list2)

print('--- Problem 1.1.1 ---') 
print(sum1)


# In[31]:


i = 0
list3 = list(range(12,101))
list4 = []
while i <= len(list3):
    list4.append(list3[i])
    i+=2
sum2 = sum(list4)

print('--- Problem 1.1.2 ---') 
print(sum2)


# In[33]:


import numpy as np 

number = int((5-3)/0.1 + 1)
array1 = np.linspace(3.0,5.0,number)
mean1 = np.mean(array1)
array2 = np.sqrt(array1)
std2 = np.std(array2)

print('--- Problem 1.2 ---')
print(mean1) 
print(std2)


# In[34]:


import numpy as np 

array3 = np.array(range(1,11))
array4 = np.array(range(11,31,2))
sum3 = np.dot(array3, array4)

print('--- Problem 1.3 ---') 
print(sum3)


# In[35]:


import numpy as np 

mu1 = 0
sigma1 = np.sqrt(3)

array5 = np.random.normal(mu1, sigma1, 1000)
mean2 = np.mean(array5)
std3 = np.std(array5)

print('--- Problem 1.4 ---') 
print(mean2)
print(std3)


# In[37]:


import numpy as np

mu2 = 0
sigma2 = 2

np.random.seed(1)
array6 = np.random.normal(mu2, sigma2, 100)



list5 = []
for i in array6:
    if i >= 0:
        list5.append(i)
    else:
        list5.append(-i)

array7 = np.array(list5)

print('--- Problem 1.5 ---') 
print(np.mean(array6))
print(np.mean(array7))


# In[39]:


import numpy as np

mu2 = 0
sigma2 = 2

np.random.seed(1)
array8 = np.random.normal(mu2, sigma2, 100)

print('--- Problem 1.6 ---') 
print(np.mean(array8))

array8[array8 < 0] = -array8[array8 < 0] 

print(np.mean(array8))


# In[40]:


def fibonacci(n):
    list1 = [1,2]
    i = 3
    while True:
        x = sum(list1)
        list1.append(x)
        if i < n:
            i += 1
        else:
            break
    return list1

print('--- Problem 2 ---') 
print(fibonacci(15))


# In[41]:


import numpy as np

matrix1 = np.matrix([[1,2,3,4,5],[0.01, 0.003, 0.015, 0.026, 0.006]])
matrix2 = matrix1.transpose()

print('--- Problem 2.1 ---') 
print(matrix2)


# In[42]:


import numpy as np

daily_return = np.array([0.01, 0.003, 0.015, 0.026, 0.006]) + 1

i = 0 
list7 = []
list7.append(daily_return[i])

while True:
    i += 1
    result = daily_return[i] * list7[i-1]
    list7.append(result)
    if i >= (daily_return.size - 1):
        break

accumulate_return = np.array(list7) - 1

print('--- Problem 2.2 ---') 
print(accumulate_return)


# In[43]:


import numpy as np
import matplotlib.pyplot as plt 

x = np.array([1,2,3,4,5])
y = accumulate_return

plt.plot(x, y, color = 'blue', linewidth = 2)
plt.xlabel('day')
plt.ylabel('cumulative return')

print('--- Problem 2.3 ---') 
plt.show()


# In[ ]:




