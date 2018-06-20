"""
Name        : Data Analysis
Author      : Hoo
Date        : May/21/2018
Version     : 0.1  base data analysis
Web_address : https://bigdata.bupt.edu.cn/
"""


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt

# 读取文件
data = pd.read_csv( 'C:/Users/Sever/Desktop/sub_stacking_2.csv', encoding='utf-8', index_col='time_stamp' )
#data_2 = pd.read_csv( 'C:/Users/Sever/Desktop/(2).csv', encoding='utf-8', index_col='time_stamp' )

#frames = [data_1, data_2]
#data = pd.concat(frames)

data.index = pd.to_datetime(data.index)
# data['loc_id'] = data['loc_id'].astype('int')
# data['num_of_people'] = data['num_of_people'].astype('int')
#print(data.info())


# 按地区功能划分
address_laboratory = [ 14, 15, 16, 20, 22, 24, 27, 33]
address_restaurant = [ 8, 10, 12, 29 ]
address_dormitory = [ 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17, 18, 19, 21, 23, 25, 26, 28, 30, 31, 32 ]


f = plt.figure(figsize=(50,30), dpi=300, facecolor='white')
plt.xlabel("day & hour")


for i in range(1, 34):
#for i in address_laboratory:
    # data_ser =  ((data[data['loc_id'] == int(i)])['number_of_people'])['2017-10-16':'2017-10-29']
    data_ser =  ((data[data['loc_id'] == int(i)])['num_of_people'])
    data_ser.plot(label = str(i))

# plt.savefig("*.jpg")  
plt.legend()
plt.show()
