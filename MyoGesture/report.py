from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import random

import my_io


OUTCOM = "./MyoGesture/NN-Outputs/best-gif/"
LOG = OUTCOM + 'Log/'
GRAPH = OUTCOM + 'Graph/'
NEURALNET_NAME = 'best-gif'

'''
Class made for ploting training, testing loss and accuracy when the method fit
(training) is finished
'''
class PlotLossesAccuracy(object):
    def __init__(self, figsize=(16, 12), dpi=200):
        plt.plot([], [])
        self.cont = 1


    def loss_plot(self, name, train_loss, valid_loss):
        plt.gca().cla()

        plt.title("Evolución del loss")
        plt.xlabel('Número de iteraciones')
        plt.ylabel('Valor del error')

        plt.grid(True)

        # plt.xlim(0, len(train_loss))
        plt.xlim(0, 500)
        plt.ylim(0., 2.51)

        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="test")

        plt.legend(loc='best', framealpha=0.5, prop={'size':'small'})
        plt.savefig(GRAPH + 'fig-loss'+name+str(self.cont)+'.png')


    def acc_plot(self,name, acc):
        plt.clf()

        plt.title('Evolución de la precisión')
        plt.xlabel('Número de iteraciones')
        plt.ylabel('Tasa de aciertos')

        plt.grid(True)

        #plt.xlim(0, len(acc))
        plt.xlim(0, 500)
        plt.ylim(0., 1.)

        plt.plot(acc, label="accuracy")
        plt.legend(loc='best', framealpha=0.5, prop={'size':'small'})

        plt.savefig(GRAPH + 'fig-acc'+name+str(self.cont)+'.png')


    def __call__(self, nn, train_history):
        name = str(nn.get_params()['dropout0_p']) + '-'
        name += str(nn.get_params()['dropout1_p']) + '-'
        name += str(nn.get_params()['dropout2_p']) + '-'
        name += str(nn.get_params()['update_learning_rate'])
        # name += str(nn.get_params()['update_momentum']) + '-'
        # name += str(nn.get_params()['dense0_nonlinearity']).split(sep=' ')[1] + '-'
        # name += str(nn.get_params()['dense1_nonlinearity']).split(sep=' ')[1] + '-'
        #name += str(nn.get_params()['dense2_nonlinearity']).split(sep=' ')[1] + str(random.randint(0, 100))

        # name = str(nn.get_params()['update']).split(sep=' ')[1] + str(random.randint(0, 100))
        # name = NEURALNET_NAME


        train_loss = np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nn.train_history_])
        acc = np.array([i["valid_accuracy"] for i in nn.train_history_])

        self.loss_plot(name, train_loss, valid_loss)
        self.acc_plot(name, acc)

        my_io.saveIterableTo([nn, nn.train_history_], LOG+name+'.log')
        my_io.list_names_parameters(OUTCOM+'name_params', name)

        self.cont+=1
        print('Guardado el log, loss y accuracy de ', name)


class Estimator(object):
    def __init__(self, whereIs = '', accuracy=0., epoch=0):
        self.whereIs = whereIs
        self.accuracy = accuracy
        self.epoch = epoch

    def __str__(self):
        return 'File:' + self.whereIs + '\nAccuracy:' + str(self.accuracy) + '\nEpoch:' + str(self.epoch)


def top10(file_with_names_of_logs):
    def update_list(t10):
        t10.sort(key=lambda estimator:estimator.accuracy, reverse=True)
        return t10

    t10 = []
    names = my_io.readFile(file_with_names_of_logs)
    for f_log in names:
        all_dict_log = my_io.get_log(LOG+f_log+'.log')

        for dict_log in all_dict_log:
            acc = dict_log['valid_accuracy']
            ep = dict_log['epoch']

            if len(t10) < 10:
                t10.append(Estimator(f_log, acc, ep))
                t10 = update_list(t10)
            else:
                if acc > t10[-1].accuracy:
                    if acc == t10[-1].accuracy:
                        if ep < t10[-1].ep:
                            t10[-1] = Estimator(f_log, acc, ep)
                            t10 = update_list(t10)
                    else:
                        t10[-1] = Estimator(f_log, acc, ep)
                        t10 = update_list(t10)
    return t10


def clasification_report(y_val, y_predicted):
    y_true = y_val
    y_pred = y_predicted
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    cr = classification_report(y_true, y_pred, target_names=target_names)
    print(cr)
    my_io.saveTo(cr, OUTCOM+NEURALNET_NAME)



def plot_confusion_matrix(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    plt.matshow(cm)
    plt.title('Matriz de confusión')
    plt.colorbar()
    plt.ylabel('Etiqueta correcta')
    plt.xlabel('Etiqueta predicha')
    # plt.show()
    plt.savefig(OUTCOM+NEURALNET_NAME+'.png', dpi=200)


def score_report(score):
    my_io.saveIterableTo(score, OUTCOM + 'score-'+NEURALNET_NAME+'.log')

def save_model(model):
    my_io.save_model(model, OUTCOM + NEURALNET_NAME)

def grid_search_scores_report(scores):
    my_io.saveIterableTo(scores, OUTCOM + 'grid_scores.log')

def grid_score_to_latex_table():
    scores = my_io.read_file('./MyoGesture/NN-Outputs/LearningR-Dropout/grid_scores.log')
    latex_table = []
    best = []

    for s in scores:
        s = s.split()
        # updates
        # line = s[7] + ' & ' + s[17] + ' & ' + s[12] + ' & ' + str(round(100*eval(s[1][:-1]),2)) + ' & '+ s[3][:-1] + '   \\' + '\\'
        line = s[10][:-1] + ' & ' + s[6][:-1] + ' & ' + s[8][:-1] + ' & ' + s[12][:-1] + ' & ' + str(round(100*eval(s[1][:-1]),2)) + '  \\' + '\\'
        latex_table.append(line)
        best.append(eval(s[1][:-1]))
#        print(line)
    print(max(best))
    my_io.saveIterableTo(latex_table, './MyoGesture/NN-Outputs/LearningR-Dropout/grid_scores_latex')
