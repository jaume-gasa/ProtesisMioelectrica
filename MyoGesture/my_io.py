from collections import deque
import numpy as np
import pickle


'''
    MODEL
'''
# Save the classifier
def save_model(m, file_name):

    with open(file_name+'.pkl', 'wb') as fid:
        pickle.dump(m, fid)

# Load classifier
def load_model(path_to_model):
    with open(path_to_model, 'rb') as fid:
        model = pickle.load(fid)
    return model


'''
    TRIVIAL INPUT/OUTPUT
'''

def saveTo(file_to_save, path):
    with open(path, mode='w+') as f:
        f.write(file_to_save)

def saveIterableTo(it, fileName):
    with open(fileName, mode='w+') as f:
        for i in it:
            if type(i) == deque:
                while i:
                    f.write(str(i.popleft())+'\n')
            else:
                f.write(str(i)+'\n')

def read_file(fileName):
    l = []
    with open(fileName, mode='r') as f:
        for line in f:
            l.append(line[:-1])
    return l


def read_my_data(file_name):
    dataList = readFile(file_name)
    for data in dataList:
        pass

def join_all_data():
    data_num = range(4, 97)  # From 4 to 96
    label = range(0, 6)  # Labels are 0, 1, 2, 3, 4, 5

    with open('./Data/all_data', mode='w+') as f_data:
        for l in label:
            for n in data_num:
                with open('./Data/data_'+str(n), mode='r') as f:
                    for line in f:
                        if line[:1] == str(l):
                            f_data.write(line)

def save_dict_to(d, fileName):
    with open(fileName, mode='w') as f:
        for key, value in sorted(d.items()):
            for v in value:
                f.write(str(key) + ', ' + str(v) + '\n')

'''
    Report
'''

def get_log(fileName):
    with open(fileName, mode='r') as f:
         for i, line in enumerate(f):
            if i is 27 - 1:
                # Convert str to a list of dict with the logs of each epoch
                return eval(line)


def list_names_parameters(fileName, name_params):
    with open(fileName, mode='a') as f:
        f.write(name_params+'\n')
