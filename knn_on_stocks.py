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

def dist(sample_1,sample_2,covar):
    diff = (sample_1[:-1] - sample_2[:-1])
    covar_I = np.linalg.inv(covar)
    dst = diff.dot(covar_I).dot(diff.T)
    return dst

def euclid_distance(x1,x2):
    e_distance = 0.0
    for i in range(len(x1)-1):
        e_distance +=(x1[i]-x2[i])**2
    return math.sqrt(e_distance)

def get_neighbours(training,testing,k,n):
    distances = []
    tf_data = training.T
    tf_data = tf_data[:-1]
    covar = np.cov(tf_data)
    for x in range(n):
        dst = dist(testing,training[x],covar)
        distances.append((training[x],dst))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def get_results(neighbors):
    classvotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classvotes:
            classvotes[response] += 1
        else:
            classvotes[response] = 1
    sortedvotes = sorted(classvotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedvotes[0][0]

def get_accuracy(testing,predictions):
    knn_correct = 0
    guess_correct = 0
    for x in range(len(testing)-1):
        if testing[x][-1] == predictions[x]:
            knn_correct += 1
        if random.randint(0,1) == testing[x][-1]:
            guess_correct += 1
    tst = stats.chisquare([knn_correct,len(testing)-1-knn_correct],f_exp=[guess_correct,len(testing)-1-guess_correct])
    p = tst[1]
    return (knn_correct/float(len(testing)-1)) * 100.0, (guess_correct/float(len(testing)-1)) * 100.0, p

def K_algorithm(stock):
    sp = stock
   # print(sp)
    data = sp.assign(foresee = 0)
    sp.index = np.arange(1, len(sp) + 1)
    for i in range(len(sp)-1):
        j = i + 1
        dclose = sp._get_value(j,'Close')
        dopen = sp._get_value(j,'Open')
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
    def mad(x): return np.fabs(x - x.mean()).mean()
    data["k6"] = data["Close"].rolling(window=5).apply(mad)

    data = data.loc[20:,["k1","k2","k3","k4","k5","k6","foresee"]]
    redata = data[:-1]
    training_data = redata.sample(frac=0.7)
    tst = list(data.index)
    for i in list(training_data.index):
        tst.remove(i)
    testing_data = data[data.index.isin(tst)]
    training_data = training_data.values
    testing_data = testing_data.values


    predictions=[]
    for x in range(len(testing_data)):
        t = x + k
        neighbors = get_neighbours(training_data,testing_data[x],k,t)
        result = get_results(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testing_data[x][-1]))
    knn_accuracy, guess_accuracy, p_value = get_accuracy(testing_data, predictions)
    print('k = ' + str(k))

    print('Accuracy: ' + repr(knn_accuracy) + '%')
    print('Matched_Accuracy: ' + repr(guess_accuracy) + '%')
    print("P_value: " + repr(p_value))
    if predictions[-1] == 1:
        print('buy in')
        return "buy in"
    else:
        print('sell out')
        return "buy out" 


today = date.today()
three_months = today - relativedelta(days=90)

url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
Symbols = companies['Symbol'].tolist()


def main(k,index=False):
    stock_final = pd.DataFrame()
    for i in Symbols:
        if i=="AAIT" or i == "AAL":
            print( str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)
            try:
                stock = []
                stock = yf.download(i,start=three_months, end=today, threads=True, progress=False)
                if len(stock) == 0:
                    None
                else:

                    stock_final = stock_final.append(stock,sort=False)
                    K_algorithm(stock_final)
                    stock_final = pd.DataFrame()
            except Exception:
                None
        else:
            break

main(k,index=True)
