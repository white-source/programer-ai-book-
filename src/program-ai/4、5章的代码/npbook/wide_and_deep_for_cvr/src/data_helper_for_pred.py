import pandas as pd
import numpy as np
import csv



def load_data_and_labels(path):
    data = []
    y = []

    count = 0
    with open(path, 'r') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for row in rdr:
            if count == 0:
                count += 1
                continue
            row_data = []
            item_emb_val = row[0].split(';')
            item_emb_val_f = [float(i) for i in item_emb_val]

            user_emb_val = row[1].split(';')
            user_emb_val_f = [float(i) for i in user_emb_val]

            row_data.extend(item_emb_val_f)
            row_data.extend(user_emb_val_f)
            row_data.append(row[2])
            row_data.append(row[3])
            row_data.append(row[4])
            row_data.append(row[5])
            row_data.append(row[6])
            data.append(row_data)
            y.append(float(row[7]))
    data = np.asarray(data)
    labels = np.asarray(y).reshape(len(y), 1)
    return data


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    load_data_and_labels("data/train.csv")