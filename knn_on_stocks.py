import numpy as np
import pandas as pd
import yfinance as yf
import math
import operator
import random
from scipy import stats
import requests
import io
from datetime import date
from dateutil.relativedelta import relativedelta

k = 7

def euclid_distance(x1,x2):
    euclid_distance = 0.0
    for i in range(len(x1)-1):
        euclid_distance += (x1[i]-x2[i])**2
    return math.sqrt(euclid_distance)

def get_neighbours(training,testing,k,n):
    distances = []
    for x in range(n):
        distance = euclid_distance(testing,training[x])
        distances.append((training[x],distance))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def get_results(neighbors):
    forecasts = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in forecasts:
            forecasts[response] += 1
        else:
            forecasts[response] = 1
    sorted_forecasts = sorted(forecasts.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_forecasts[0][0]

def get_accuracy(testing,predictions):
    knn_accuracy = 0
    for x in range(len(testing)-1):
        if testing[x][-1] == predictions[x]:
            knn_accuracy += 1
    tests = stats.chisquare([knn_accuracy,len(testing)-1-knn_accuracy])
    p = tests[1]
    return (knn_accuracy/float(len(testing)-1)) * 100.0, p

#compute the absolute data of values in x
def mad(x): 
    return np.fabs(x - x.mean()).mean()

def K_algorithm(stock):
    data = stock.assign(foresee = 0)
    stock.index = np.arange(1, len(stock) + 1)
    for i in range(len(stock)-1):
        j = i + 1
        dclose = stock._get_value(j,'Close')
        dopen = stock._get_value(j,'Open')
        if dclose - dopen > 0:
            data._set_value(i,'foresee',1)
        else:
            data._set_value(i,'foresee',0)
    data["k1"] = (data["High"] - data["Low"])/data["Open"]
    data["k2"] = (data["Close"] - data["Open"])/data["Open"]
    data["k3"] = data["Volume"]/100000
    data["5d"] = data["Close"].rolling(window=5).mean()
    data["13d"] = data["Close"].rolling(window=13).mean()
    data["20d"] = data["Close"].rolling(window=20).mean()
    data["k4"] = (data["5d"] - data["20d"])
    data["k5"] = (data["5d"] - data["13d"])
    data["k6"] = data["Close"].rolling(window=5).apply(mad)

    data = data.loc[20:,["k1","k2","k3","k4","k5","k6","foresee"]]
    redata = data[:-1]
    training_data = redata.sample(frac=0.7)
    test = list(data.index)
    for i in list(training_data.index):
        test.remove(i)
    testing_data = data[data.index.isin(test)]
    training_data = training_data.values
    testing_data = testing_data.values

    predictions=[]
    for x in range(len(testing_data)):
        t = x + k
        neighbors = get_neighbours(training_data,testing_data[x],k,t)
        result = get_results(neighbors)
        predictions.append(result)
    knn_accuracy, p_value = get_accuracy(testing_data, predictions)
    print('KNN Accuracy: ' + repr(knn_accuracy) + '%')
    print("P_value: " + repr(p_value))

    return p_value


today = date.today()
three_months = today - relativedelta(days=90)

url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
Symbols = companies['Symbol'].tolist()


def main(k,index=False):
    stock_final = pd.DataFrame()
    prediction_list = {}
    for i in Symbols:
        if i.startswith("AA"):
            print( str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)
            try:
                stock = []
                stock = yf.download(i,start=three_months, end=today, threads=True, progress=False)
                if len(stock) == 0:
                    None
                else:
                    stock_final = stock_final.append(stock,sort=False)
                    prediction_list[i] = K_algorithm(stock_final)
                    stock_final = pd.DataFrame()
            except Exception:
                None
        else:
            break
    predictions_sorted = sorted(prediction_list.items(), key=lambda x: x[1], )
    return predictions_sorted

computations = main(k,index=True)
print("Best companies to invest: ")
for x in list(reversed(list(computations)))[0:5]:
    print (x[0])

print("The last companies you should invest: ")
for x in list(computations)[0:7]:
    print (x[0])

