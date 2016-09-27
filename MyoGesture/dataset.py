import my_io
import numpy as np
import re

from sklearn.preprocessing import scale

'''
0 -> 5085
1 -> 2373
2 -> 6696
3 -> 4648
4 -> 2565
5 -> 5173

'''


def balanced_dataset():
    X, y = load_dataset()
    X = [i for i in X]
    y = [i for i in y]
    labels = [0,1,2,3,4,5]
    dict_data = {0:[], 1:[],2:[],3:[],4:[],5:[]}

    for i in range(len(X)):
        dict_data[y[i]].append(X[i])

    for i in dict_data.keys():
        print(len(dict_data[i]))

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    cont = 2373

    for i in dict_data.keys():
        while cont > 0:
            X_train.append(dict_data[i].pop())
            y_train.append(i)
            cont-=1

        cont = len(dict_data[i])
        while cont > 0:
            X_val.append(dict_data[i].pop())
            y_val.append(i)
            cont-=1
        cont = 2373

    # 14238 14238 12302 12302
    #print(len(X_train), len(y_train),  len(X_val), len(y_val))
    # for x, y in zip(X_train, y_train):
    #    print(x, y)

    X = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float64)
    y_val = np.array(y_val, dtype=np.int32)

    # print(X[0], type(X[0]), type(X[0][0]))
    # X = preprocessing(X)
    # print(X[0], type(X[0]), type(X[0][0]))

    return preprocessing(X), y, preprocessing(X_val), y_val


def balanced_validationset():
    X, y = load_dataset()
    X = [i for i in X]
    y = [i for i in y]
    X_val = []
    y_val = []
    labels = [0,1,2,3,4,5]
    count = 333

    for i in labels:
        for j in range(len(y)):
            if i == y[j]:
                X_val.append(X[j])
                y_val.append(y[j])
                X.pop(j)
                y.pop(j)
                count-=1
                if count == 0:
                    count = 333
                    break
    # Take the last two to get round numbers on validation and data test
    # Validation is 2000 and training is 24540
    X_val.append(X.pop())
    X_val.append(X.pop())
    y_val.append(y.pop())
    y_val.append(y.pop())

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    return preprocessing(X), y, preprocessing(X_val), y_val


def load_dataset():
    list_data = my_io.read_file('./Data/all_data')
    X = []
    y = []

    for data in list_data:
        data = re.findall(r'\d+', data)
        y.append(eval(data[0]))
        X.append([eval(emg) for emg in data[1:]])

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    # Declare a random number generator
    rng = np.random.RandomState(seed=0)

    # Shuffle the data the right way to avoid missleading label-emg
    permutation = rng.permutation(len(X))
    X, y = X[permutation], y[permutation]

    # X = np.array(scale(X), dtype=np.float64)

    # return all the arrays in order, as expected in main().
    return X, y


def load_dataset_and_validationset():
    X, y  = load_dataset()

    # We reserve the last 2000 training examples for validation.
    X_train, X_val = X[:-2000], X[-2000:]
    y_train, y_val = y[:-2000], y[-2000:]

    return X_train, y_train, X_val, y_val

def load_preprocesed_dataset_and_validationset():
    X, y = load_dataset()
    X = preprocessing(X)

    # We reserve the last 2000 training examples for validation.
    X_train, X_val = X[:-2000], X[-2000:]
    y_train, y_val = y[:-2000], y[-2000:]

    return X_train, y_train, X_val, y_val


def load_preprocesed_dataset():
    X, y = load_dataset()
    X = preprocessing(X)
    return X, y


def preprocessing(X, y=None):
    X = np.array(scale(X), dtype=np.float32)
    if y is not None:  # Probar si funciona este caso
        X = np.array(scale(X), dtype=np.float32)
        return X, y
    else:
        return X

if __name__ == '__main__':
    #balanced_trainingset()
    load_dataset()
