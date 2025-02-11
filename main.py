"""
import math #IMPORTS HERE
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
import pandas as pd
"""
import csv
#----------------------------------------------------------
n2o = [];
humidity=[];
temperature=[];
pression=[];
# -------------*------------------------*-----------------
def main(n2o, humidity, temperature, pression):
    print('oxido nitroso:')
    print(n2o)
    print('\n humedad:')
    print(humidity)
    print('\n temperatura:')
    print(temperature)
    print('\n presion:')
    print(pression)

# to read the db on the csv field
# -------------*------------------------*-----------------
if __name__ == '__main__': 
    with open("datos.csv", newline='') as csvFile:
        spamreader = csv.reader(csvFile, delimiter=' ', quotechar=' ')
        i = 0
        for row in spamreader:
            if i > 0:
                #xList.append(int(data[0]))
                #OxidoNitroso(y[0]);Humedad(x1[1]);Temperatura(x2[2]);Presion(x3[3])
                data = row[0].split(',')
                n2o.append(float(data[0]))
                humidity.append(float(data[1]))
                temperature.append(float(data[2]))
                pression.append(float(data[3]))
            i += 1;
    main(n2o, humidity, temperature, pression);

    
# -------------*------------------------*-----------------
# list(map(smt, smt2))