import pandas as pd
import numpy as np
import csv

def load_data_and_labels(path):
    data = []
    y = []
    total_q = []

    # count = 0
    with open(path, 'r') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for row in rdr:
            y.append(float(row[1]))


    # data = np.asarray(data)
    total_q = np.asarray(total_q)
    y = np.asarray(y)
    return data, y


data, y = load_data_and_labels('../data/zutao2.csv')

bins = pd.qcut(y, 50, retbins=True)
print(bins[0])

