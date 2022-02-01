'''

'''
import numpy as np
import tensorflow.keras.backend as K
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import *
from datetime import datetime
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import load_model


def selectsample(model, x_test, y_test, delta, iterate):
    test = x_test
    output = model.predict(test)
    max_output = np.max(output, axis=1)
    index = np.argsort(max_output)
    # devide into 3 strata
    index1 = index[0:int(x_test.shape[0] * 0.1)]
    index2 = index[int(x_test.shape[0] * 0.1):int(x_test.shape[0] * 0.2)]
    index3 = index[int(x_test.shape[0] * 0.2):]

    train = x_train
    output_train = model.predict(train)
    max_output_train = np.max(output_train, axis=1)
    index1_train = []
    index2_train = []
    index3_train = []
    for i in range(len(max_output_train)):
        if max_output_train[i] >= max_output[index1[0]] and max_output_train[i] <= max_output[index1[-1]]:
            index1_train.append(i)
        elif max_output_train[i] >= max_output[index2[0]] and max_output_train[i] <= max_output[index2[-1]]:
            index2_train.append(i)
        elif max_output_train[i] >= max_output[index3[0]] and max_output_train[i] <= max_output[index3[-1]]:
            index3_train.append(i)

    print(len(index1_train))
    print(len(index2_train))
    print(len(index3_train))

    label = y_train[index1_train]
    pred = np.argmax(output_train[index1_train], axis=1)
    s1 = np.std(pred == label)
    print("strata1_Sh", s1)
    acc_1 = np.sum(pred == label) / len(pred)
    print("strata1_acc", acc_1)

    label = y_train[index2_train]
    pred = np.argmax(output_train[index2_train], axis=1)
    s2 = np.std(pred == label)
    print("strata2_Sh", s2)
    acc_2 = np.sum(pred == label) / len(pred)
    print("strata2_acc", acc_2)

    label = y_train[index3_train]
    pred = np.argmax(output_train[index3_train], axis=1)
    s3 = np.std(pred == label)
    print("strata3_Sh", s3)
    acc_3 = np.sum(pred == label) / len(pred)
    print("strata3_acc", acc_3)

    total_ws = len(index1_train) * s1 + len(index2_train) * s2 + len(index3_train) * s3
    w = [len(index1_train) * s1 / total_ws, len(index2_train) * s2 / total_ws, len(index3_train) * s3 / total_ws]
    print(w)
    print(sum(w))

    acc_list1 = []
    acc_list2 = []

    select_num = 50
    for i in range(iterate):
        if w[0] == 0:
            acc_1 = 1
        else:
            arr = np.random.permutation(index1.shape[0])
            temp_index = arr[0:int(select_num * w[0])]
            statra_index1 = index1[temp_index]
            statra_index1 = statra_index1.astype('int')
            label = y_test[statra_index1]
            orig_sample = x_test[statra_index1]
            orig_sample = orig_sample.reshape(-1, 32, 32, 3)
            pred = np.argmax(model.predict(orig_sample), axis=1)
            acc_1 = np.sum(pred == label) / orig_sample.shape[0]

        if w[1] == 0:
            acc_2 = 1
        else:
            arr = np.random.permutation(index2.shape[0])
            temp_index = arr[0:int(select_num * w[1])]
            statra_index2 = index2[temp_index]
            statra_index2 = statra_index2.astype('int')
            label = y_test[statra_index2]
            orig_sample = x_test[statra_index2]
            orig_sample = orig_sample.reshape(-1, 32, 32, 3)
            pred = np.argmax(model.predict(orig_sample), axis=1)
            acc_2 = np.sum(pred == label) / orig_sample.shape[0]

        if w[2] == 0:
            acc_3 = 1
        else:
            arr = np.random.permutation(index3.shape[0])
            temp_index = arr[0:int(select_num * w[2])]
            statra_index3 = index3[temp_index]
            statra_index3 = statra_index3.astype('int')
            label = y_test[statra_index3]
            orig_sample = x_test[statra_index3]
            orig_sample = orig_sample.reshape(-1, 32, 32, 3)
            pred = np.argmax(model.predict(orig_sample), axis=1)
            acc_3 = np.sum(pred == label) / orig_sample.shape[0]

        arr = np.random.permutation(x_test.shape[0])
        random_index = arr[0:select_num]
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

    for k in range(50):
        print("the {} exp".format(k))
        acc1, acc2 = experiments(delta=10, iterate=97)
        np.savetxt('ssoa_m_t/select{}.csv'.format(k), acc1)
        np.savetxt('ssoa_m_t/random{}.csv'.format(k), acc2)
        elapsed = (datetime.now() - start_time)
        print("Time used: ", elapsed)

    elapsed = (datetime.now() - start_time)
    print("Time used: ", elapsed)
