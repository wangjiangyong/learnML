# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
############################## concat()用法
# 用于将2个DataFrame连接到一起，一般是将训练集与测试集连接在一起对属性值进行预处理。
# 这样处理是可以的，因为对属性值的处理一般要求数值越多越好，而且对训练集与测试集的数据预处理要求要是一样的。
# 只要在训练模型的时候讲2者分开就行。
# 相同列合并到一起，不同时新建列，缺省值为NaN

df1 = pd.DataFrame({'a':[1,2,3],
                    'b':[4,5,6]})

df2 = pd.DataFrame({'a':['a','b','c'],
                    'c':['d','e','f']})
#print df1,'\n',df2

df12 = pd.concat([df1,df2])
#print df12

df12.reset_index(inplace=True)
#print df12

df12.drop('index',axis=1,inplace=True)
#print df12

df12 = df12.reindex_axis(df1.columns,axis=1)
#print df12

#print df12.shape #输出行列数

####################### map()用法
df3 = pd.DataFrame(np.arange(8).reshape(4,2),columns=['a','b'])
#print df3
# Series 或 DataFrame 的列都可以调用一个map()方法。
# 该方法接受一个函数或字典作为参数，并将之应用于该列的每一个元素，将元素值映射为另一个值。
# 多用于数据离散化。
# lambda函数也可以用自定义函数实现
df3['c'] = df3['a'].map(lambda x:x+1)
# df3['a'] = df3['a'].map(...) 也可以修改原来列的值
#print df3

# 当元素值少时，可以直接用字典修改：
df4 = pd.DataFrame({'a':['female','male'],'b':['male','Female']})
df4['a'] = df4['a'].map({'female':0,'male':1})
#print df4

################################# isin()用法
# 判断某一列元素是否属于列表里的元素，返回True False列表。
# 如果为True，则对该行元素进行操作，False时不操作

df5 = pd.DataFrame({'columns1':['a','b','c'],'columns2':['c','d','e']})
df5.columns1[df5.columns1.isin(['a','b'])] = 'cc'
#print df5

################################  采用中位数/出现次数设置missing值
df6 = pd.DataFrame(np.arange(8).reshape(4,2),columns=['a','b'])
df7 = pd.DataFrame(np.arange(8).reshape(4,2),columns=['c','b'])
df67 = pd.concat([df6,df7])
df67.reset_index(inplace=True)
df67.drop('index',axis=1,inplace=True)
#df67.a[df67.a.isnull()] = df67.a.dropna().median()  #??????????????????
# SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame
#print df67


df8 = pd.DataFrame({'a':['a','b','a'],
                    'c':['e','f','g']})
#print df8
freq_max = df8.a.dropna().mode().values  #对于一列非数字，例如字符，要找到出现频率最高的字符
#print "freq_max:" , freq_max

###################### 属性数字化---枚举
# 某一属性，其值表示为字符，且范围较少时，可选择使用枚举进行数字化
# 用np.unique()生成唯一化后的元素，在用enumerate()生成对应元组，转化为列表后生成字典。
# 再对字典进行map操作，即可实现数值化。
df9 = pd.DataFrame({'aa':['a','b','c'],
                    'dd':['d','e','f']})

# np.unique()去除重复元素
# enumerate()是python的内置函数.对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值.
unique_value = list(enumerate(np.unique(df9.aa))) #[(0, 'a'), (1, 'b'), (2, 'c')]
dict = {key:value for value,key in unique_value} # 构建字符/数字映射的字典。
df9.aa = df9.aa.map(lambda x:dict[x]).astype(int) # 用数字替换字符。
#print df9


############################ dummy变量
# 作用条件与枚举类似，属性值范围不大时，会为每一个值生成新的一列。结果需要concat
# 每个属性值对应一列，所以属性值很多的情况下不适用，会生成较大的df
dfA = pd.DataFrame({'column1':['aa','bb','cc'] ,
                    'column2':['dd','ee','ff']})
dummy_dfA_column1 = pd.get_dummies(dfA.column1)  #DataFrame类型,会为每一个值生成新的一列
dummy_dfA_column1 = dummy_dfA_column1.rename(columns=lambda x:'dummy_'+str(x))
dfA = pd.concat([dfA,dummy_dfA_column1],axis=1) #将dummy变量变换后的新的属性列直接concat到原来的df中即可。
#print dfA


########################### loc()函数
# 对应列选取多行，第一个元素取行，第二个元素对应列，默认情况下为所有列
dfB = pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
#print dfB.loc[(dfB.a.values > 2)]
#print dfB.loc[(dfB.a.values > 2),'a'] #只作用于a列，输出a列
dfB.loc[(dfB.a.values > 2),'a'] = 2  #对其赋值，则改变df的值
#print dfB


##############################pandas.qcut()
# 将数值属性划分成几个bin，这是一种连续数据离散化的处理方式。
# 使用函数来离散化连续数据，它使用分位数对数据进行划分，可以得到大小基本相等的bin。

##############################factorize()
#主要是将列表中字母值用枚举表示，相同值用同一数字。结果只生成一列，可以在原来列中操作
dfC  = pd.DataFrame({'column1':['a','b','a'],
                     'column2':['e','f','g']})
#print pd.factorize(dfC.column1)
dfC.column1 = pd.factorize(dfC.column1)[0]
dfC.column2 = pd.factorize(dfC.column2)[0]

#print dfC

#####################################scaler()
#规范化，把数据压缩到一个范围

dfD = pd.DataFrame({'aa':[5,10,15,20],
                    'bb':[0,3,6,9],
                   'cc':[1000,0,500,50],
                    'dd':[1,100,3,2]})

scaler = preprocessing.StandardScaler()

dfD = scaler.fit_transform(dfD)
#print dfD



