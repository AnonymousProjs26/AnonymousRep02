'''

'''
import numpy as np
import tensorflow.keras.backend as K
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import *
import math
from datetime import datetime
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model

def selectsample(model, x_test, y_test, delta, iterate):
    global index3, index1, index2, index1_adaptive, index2_adaptive, index3_adaptive, total_adaptive_sampling, w
    acc_list1 = []
    acc_list2 = []

    select_num = 50 - total_adaptive_sampling
    for i in range(iterate):
        if w[0] == 0 or math.ceil(select_num * w[0]) < 1:
            acc_1 = 1
        else:
            index1 = np.setdiff1d(index1, index1_adaptive, True)
            arr = np.random.permutation(index1.shape[0])
            temp_index = arr[0:math.ceil(select_num * w[0])]
            statra_index1 = index1[temp_index]
            statra_index1 = statra_index1.astype('int')

            statra_index1 = np.union1d(statra_index1, index1_adaptive)
            print("0: ", len(statra_index1))
            label = y_test[statra_index1]
            orig_sample = x_test[statra_index1]
            orig_sample = orig_sample.reshape(-1, 32, 32, 3)
            pred = np.argmax(model.predict(orig_sample), axis=1)
            acc_1 = np.sum(pred == label) / orig_sample.shape[0]


        if w[1] == 0 or math.ceil(select_num * w[1]) < 1:
            acc_2 = 1
        else:
            index2 = np.setdiff1d(index2, index2_adaptive, True)
            arr = np.random.permutation(index2.shape[0])
            temp_index = arr[0:math.ceil(select_num * w[1])]
            statra_index2 = index2[temp_index]
            statra_index2 = statra_index2.astype('int')

            statra_index2 = np.union1d(statra_index2, index2_adaptive)
            print("1: ", len(statra_index2))
            label = y_test[statra_index2]
            orig_sample = x_test[statra_index2]
            orig_sample = orig_sample.reshape(-1, 32, 32, 3)
            pred = np.argmax(model.predict(orig_sample), axis=1)
            acc_2 = np.sum(pred == label) / orig_sample.shape[0]

        if w[2] == 0 or math.ceil(select_num * w[2]) < 1:
            acc_3 = 1
        else:
            index3 = np.setdiff1d(index3, index3_adaptive, True)
            arr = np.random.permutation(index3.shape[0])
            temp_index = arr[0:math.ceil(select_num * w[2])]
            statra_index3 = index3[temp_index]
            statra_index3 = statra_index3.astype('int')

            statra_index3 = np.union1d(statra_index3, index3_adaptive)
            print("2: ", len(statra_index3))
            label = y_test[statra_index3]
            orig_sample = x_test[statra_index3]
            orig_sample = orig_sample.reshape(-1, 32, 32, 3)
            pred = np.argmax(model.predict(orig_sample), axis=1)
            acc_3 = np.sum(pred == label) / orig_sample.shape[0]

        arr = np.random.permutation(x_test.shape[0])
        random_index = arr[0:select_num+total_adaptive_sampling]
        random_index = random_index.astype('int')

        acc1 = 0.1 * acc_1 + 0.1 * acc_2 + 0.8 * acc_3
        acc_list1.append(acc1)

        label = y_test[random_index]
        orig_sample = x_test[random_index]
        orig_sample = orig_sample.reshape(-1, 32, 32, 3)
        pred = np.argmax(model.predict(orig_sample), axis=1)
        acc2 = np.sum(pred == label) / orig_sample.shape[0]
        acc_list2.append(acc2)

        print("numuber of samples is {!s}, select acc is {!s}, random acc is {!s}".format(
            orig_sample.shape[0], acc1, acc2))

        select_num = select_num + delta

    return acc_list1, acc_list2


def experiments(delta, iterate):
    pred = np.argmax(model.predict(x_test), axis=1)
    true_acc = np.sum(pred == y_test) / x_test.shape[0]
    print("The final acc is {!s}".format(true_acc))

    acc_list1, acc_list2 = selectsample(
        model, x_test, y_test, delta, iterate)
    print("the select acc std is {!s}".format(
        np.mean(np.abs(acc_list1 - true_acc))))
    print("the random acc std is {!s}".format(
        np.mean(np.abs(acc_list2 - true_acc))))

    return np.array(acc_list1), np.array(acc_list2)


if __name__ == '__main__':
    start_time = datetime.now()

    # Load the data.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # preprocess the data set
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    print(mean)
    print(std)
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 100)
    y_train = np.argmax(y_train, axis=1)
    y_test = keras.utils.to_categorical(y_test, 100)
    y_test = np.argmax(y_test, axis=1)

    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_test.shape)

    # Load the model.
    model = load_model('cifar100vgg.h5')

    test = x_test
    output = model.predict(test)
    max_output = np.max(output, axis=1)
    index = np.argsort(max_output)
    # devide into 3 strata
    index1 = index[0:int(x_test.shape[0] * 0.1)]
    index2 = index[int(x_test.shape[0] * 0.1):int(x_test.shape[0] * 0.2)]
    index3 = index[int(x_test.shape[0] * 0.2):]

    total_adaptive_sampling = 30
    np.random.seed(6)

    arr = np.random.permutation(index1)
    random_index = arr[0:10]
    random_index = random_index.astype('int')
    index1_adaptive = random_index
    print(index1_adaptive)
    label = y_test[index1_adaptive]
    pred = np.argmax(output[index1_adaptive], axis=1)
    s1 = np.std(pred == label)
    print("strata1_Sh", s1)
    acc_1 = np.sum(pred == label) / len(pred)
    print("strata1_acc", acc_1)

    arr = np.random.permutation(index2)
    random_index = arr[0:10]
    random_index = random_index.astype('int')
    index2_adaptive = random_index
    print(index2_adaptive)
    label = y_test[index2_adaptive]
    pred = np.argmax(output[index2_adaptive], axis=1)
    s2 = np.std(pred == label)
    print("strata2_Sh", s2)
    acc_2 = np.sum(pred == label) / len(pred)
    print("strata2_acc", acc_2)

    arr = np.random.permutation(index3)
    random_index = arr[0:10]
    random_index = random_index.astype('int')
    index3_adaptive = random_index
    print(index3_adaptive)
    label = y_test[index3_adaptive]
    pred = np.argmax(output[index3_adaptive], axis=1)
    s3 = np.std(pred == label)
    print("strata3_Sh", s3)
    acc_3 = np.sum(pred == label) / len(pred)
    print("strata3_acc", acc_3)

    total_ws = len(index1) * s1 + len(index2) * s2 + len(index3) * s3
    w = [len(index1) * s1 / total_ws, len(index2) * s2 / total_ws, len(index3) * s3 / total_ws]
    print(w)
    print(sum(w))


    for k in range(50):
        print("the {} exp".format(k))
        acc1, acc2 = experiments(delta=10, iterate=96)
        np.savetxt('ssoa_m_p/select{}.csv'.format(k), acc1)
        np.savetxt('ssoa_m_p/random{}.csv'.format(k), acc2)
        elapsed = (datetime.now() - start_time)
        print("Time used: ", elapsed)

    elapsed = (datetime.now() - start_time)
    print("Time used: ", elapsed)
