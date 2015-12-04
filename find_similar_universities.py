import pandas
import numpy as np
import math

def max_feature_values(data):
    max_list = []
    data = data.drop('univName',1)
    for j in data.columns:
        max_list.append(np.array(data[j].tolist()).astype(np.float).max())
    return max_list

def min_feature_values(data):
    min_list = []
    data = data.drop('univName',1)
    for j in data.columns:
        min_list.append(np.array(data[j].tolist()).astype(np.float).min())
    return min_list

def normalize(data,max_data,min_data):
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = (data[i] - min_data[i]) / float(max_data[i] - min_data[i])
    return data

def find_mean(data,length):
    for i in range(len(data)):
        data[i] = data[i] / float(length)
    return data

def euclidean_distance(x,y):
  return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

data = pandas.read_csv('processed_data.csv')
data = data.drop('Unnamed: 0',1)
data = data.drop('Unnamed: 0.1',1)

max_list = max_feature_values(data)
min_list = min_feature_values(data)
sum_of_features_for_univ = [0] * (len(data.columns) - 1)
university_data = []
university_list = list(set(data['univName'].tolist()))
for i in range(len(university_list)):
    cur_data = data[data['univName'] == university_list[i]]
    cur_data = cur_data.drop('univName',1)
    for j in range(len(cur_data)):
        cur_row_for_univ = cur_data.iloc[j].tolist()
        cur_feature_for_univ = normalize(cur_row_for_univ,max_list,min_list)
        sum_of_features_for_univ = [x+y for x,y in zip(cur_feature_for_univ, sum_of_features_for_univ)]
    average_of_features_for_univ = find_mean(sum_of_features_for_univ,len(cur_data))
    university_data.append(average_of_features_for_univ)

similar_univs = pandas.DataFrame(columns=('univName','1','2','3','4','5','6','7','8','9'))
current_univ = []
for i in range(len(university_list)):
    dist = []
    cur_univ_list = []
    for j in range(len(university_list)):
        if i!= j:
            dist.append(euclidean_distance(university_data[i],university_data[j]))
            cur_univ_list.append(university_list[j])
    dist = np.array(dist)
    ind = np.argpartition(dist, 9)[:9]
    ind = ind.tolist()
    temp = []
    temp.append(university_list[i])
    for k in ind:
        temp.append(cur_univ_list[k])
    similar_univs.loc[i] = temp
similar_univs.to_csv('similar_universities.csv')
