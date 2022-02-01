# -*- coding: UTF-8 -*-
import time
import os
import sys
import datetime
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import random
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from numpy import arange
import hdbscan
import openpyxl
import argparse
from tensorflow.keras.applications import vgg19,resnet50
from sklearn.cluster import KMeans
import math

threshold = 0

def get_score(x_test, y_test, model):
 
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('预测错的数目：', len(x_test)*(1-score[1]))
    return score

def get_cifar10_cn12(**kwargs):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_test = X_test.astype('float32')
    X_test = (X_test / 255.0) - (1.0 - 0.5)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    X_train = X_train.astype('float32')
    X_train = (X_train / 255.0) - (1.0 - 0.5)
    Y_train = keras.utils.to_categorical(Y_train, 10)
    return X_test, Y_test, X_train, Y_train


def get_cifar10_vgg16(**kwargs):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    Y_test = keras.utils.to_categorical(Y_test, 10)
    Y_train = keras.utils.to_categorical(Y_train, 10)
    return X_test, Y_test, X_train, Y_train


def get_cifar100_vgg16(**kwargs):
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    Y_test = keras.utils.to_categorical(Y_test, 100)
    Y_train = keras.utils.to_categorical(Y_train, 100)
    return X_test, Y_test, X_train, Y_train

def get_data(exp_id):
    exp_model_dict = {'cn12': get_cifar10_cn12,
                      'cifar10_vgg16': get_cifar10_vgg16,
                      'cifar100_vgg16': get_cifar100_vgg16}
    return exp_model_dict[exp_id](exp_id=exp_id)


def get_model(exp_id):
    basedir = os.path.abspath(os.path.dirname(__file__))

    exp_model_dict = {'cn12': 'model/cifar10_cn12.h5',
                      'cifar10_vgg16': 'model/cifar10vgg.h5',
                      'cifar100_vgg16': 'model/cifar100vgg.h5'}

    if exp_id == 'vgg19':
        my_model = vgg19.VGG19(weights='imagenet')
        adam = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        my_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    elif exp_id == 'resnet50':
        my_model = resnet50.ResNet50(weights='imagenet')
        adam = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        my_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    elif exp_id in exp_model_dict.keys():
        my_model = keras.models.load_model(os.path.join(basedir,exp_model_dict[exp_id]))
    else:
        raise Exception("no such dataset found: {}".format(exp_id))

    return my_model

def get_acc(exp_id):
    acc_dict = {'cn12': 0.8064,
                'cifar10_vgg16': 0.9375,
                'cifar100_vgg16': 0.6944}
    return acc_dict[exp_id]

if __name__=="__main__":

    basedir = os.path.abspath(os.path.dirname(__file__))

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--exp_id", type=str, help="exp_identifiers")

    console_flags, unparsed = parse.parse_known_args(sys.argv[1:])

    exp_id = console_flags.exp_id
    acc = get_acc(exp_id)

    my_model = get_model(exp_id=exp_id)
    X_test,Y_test,X_train,Y_train = get_data(exp_id=exp_id)
    print(my_model.summary())
    print(X_test.shape)
    print(X_train.shape)

    start = datetime.datetime.now()

    dense_output = np.max(my_model.predict(X_test), axis=1)
    dense_output = dense_output.reshape(-1,1)
    print(dense_output)

    k_means = KMeans(n_clusters=3, random_state=10)
    k_means.fit(dense_output)

    print(k_means.cluster_centers_)
    print(k_means.inertia_)

    labels = k_means.labels_

    print(labels)
    print(np.max(labels))
    print(np.min(labels))

    y_pred_list = labels.tolist()
    countlist = []

    for i in range(np.min(labels), np.max(labels) + 1):
        countlist.append(y_pred_list.count(i))

    print(countlist)

    print(np.sort(countlist))
    print(np.argsort(countlist))

    res = {}
    for i, l in enumerate(labels):
        if l not in res:
            res[l] = []
        res[l].append(i)

    for key in res:
        X_test3 = []
        Y_test3 = []
        print(key, len(res[key]))
        for i in range(len(res[key])):
            X_test3.append(X_test[res[key][i]])
            Y_test3.append(Y_test[res[key][i]])
        X_test3 = np.array(X_test3)
        Y_test3 = np.array(Y_test3)
        score = get_score(X_test3, Y_test3, my_model)

    s_adaptive = {}
    acc_adaptive = {}
    total_ws = 0
    adaptive_index = {}
    total_adaptive_sampling = 0

    for key in res.keys():
        np.random.seed(5)
        arr = np.random.permutation(res[key])
        random_index = arr[0:10]
        random_index = random_index.astype('int')

        total_adaptive_sampling = total_adaptive_sampling + len(random_index)

        adaptive_index[key] = random_index
        label = Y_test[random_index]
        orig_sample = X_test[random_index]
        pred = np.argmax(my_model.predict(orig_sample), axis=1)
        label = np.argmax(label, axis=1)
        s_stratum = np.std(pred == label)
        print(key, "strata_Sh", s_stratum)
        acc_stratum = np.sum(pred == label) / orig_sample.shape[0]
        print(key, "strata_acc", acc_stratum)
        s_adaptive[key] = s_stratum
        acc_adaptive[key] = acc_stratum

        total_ws = total_ws + len(res[key]) * s_stratum

    w = {}

    for key in res.keys():
        w[key] = len(res[key]) * s_adaptive[key] / total_ws

    print(w)

    delta = 10
    iterate = 96

    for k in range(50):
        print("the {} exp".format(k))

        acc_list1 = []
        acc_list2 = []
        statra_index1 = []
        random_index = []
        select_num = 50 - total_adaptive_sampling
        for i in range(iterate):
            acc_test = {}
            for key in res.keys():
                print(key, math.ceil(select_num * w[key]))
                if w[key] == 0 or math.ceil(select_num * w[key]) <= 0:
                    acc_1 = 1
                    acc_test[key] = acc_1
                else:
                    index = res[key]
                    index = np.array(index)
                    index = np.setdiff1d(index, adaptive_index[key], True)
                    arr = np.random.permutation(index.shape[0])
                    temp_index = arr[0:math.ceil(select_num * w[key])]
                    statra_index1 = index[temp_index]
                    statra_index1 = statra_index1.astype('int')

                    statra_index1 = np.union1d(statra_index1, adaptive_index[key])
                    print(key, len(statra_index1))
                    label = Y_test[statra_index1]
                    orig_sample = X_test[statra_index1]
                    pred = np.argmax(my_model.predict(orig_sample), axis=1)
                    label = np.argmax(label, axis=1)
                    acc_1 = np.sum(pred == label) / orig_sample.shape[0]
                    acc_test[key] = acc_1

            arr = np.random.permutation(X_test.shape[0])
            random_index = arr[0:select_num]
            random_index = random_index.astype('int')

            acc1 = 0
            print(acc_test)
            for key in res.keys():
                acc1 = acc1 + acc_test[key] * (len(res[key]) / 10000)

            acc_list1.append(acc1)
            label = Y_test[random_index]
            orig_sample = X_test[random_index]
            pred = np.argmax(my_model.predict(orig_sample), axis=1)
            label = np.argmax(label, axis=1)
            acc2 = np.sum(pred == label) / orig_sample.shape[0]
            acc_list2.append(acc2)

            print("numuber of samples is {!s}, select acc is {!s}, random acc is {!s}".format(
                select_num, acc1, acc2))
            print(np.abs(acc1 - acc), np.abs(acc2 - acc))

            select_num = select_num + delta

        np.savetxt(os.path.join(basedir, "kmeans_adaptive_{}/select{}.csv".format(exp_id, k)), np.array(acc_list1))
        np.savetxt(os.path.join(basedir, "kmeans_adaptive_{}/random{}.csv".format(exp_id, k)), np.array(acc_list2))

        elapsed = (datetime.datetime.now() - start)
        print("Time used: ", elapsed)

    elapsed = (datetime.datetime.now() - start)
    print("Time used: ", elapsed)