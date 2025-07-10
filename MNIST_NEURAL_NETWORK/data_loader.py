import numpy as np
import csv

def load_mnist_from_csv(file_path, limit=None):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  
        data = list(reader)

    data = np.array(data, dtype=np.float32)
    labels = data[:, 0].astype(int)
    images = data[:, 1:] / 255.0  

    if limit:
        images = images[:limit]
        labels = labels[:limit]


    X = [img.reshape(784, 1) for img in images]
    y = [label_to_one_hot(lbl) for lbl in labels]

    return list(zip(X, y))

def label_to_one_hot(label):
    vec = np.zeros((10, 1))
    vec[label] = 1.0
    return vec

def load_mnist_from_csv(file_path, limit=None):
    import numpy as np
    import csv

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        data = list(reader)

    valid_data = []
    for row in data:
        if len(row) != 785:  
            continue
        valid_data.append(row)

    data = np.array(valid_data, dtype=np.float32)
    labels = data[:, 0].astype(int)
    images = data[:, 1:] / 255.0

    if limit:
        images = images[:limit]
        labels = labels[:limit]

    X = [img.reshape(784, 1) for img in images]
    y = [label_to_one_hot(lbl) for lbl in labels]

    return list(zip(X, y))
