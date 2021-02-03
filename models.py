# -*- coding: utf-8 -*-
# CreateBy: kai
# CreateAt: 2021/1/17
import myo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
from threading import Lock
import time
from collections import deque
import numpy as np
import argparse

from collect_data import data_process1, collect, engine


class Listener(myo.DeviceListener):
    def __init__(self, n, clf):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.clf = clf
        self.data = []

    def on_connected(self, event):
        print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)

    def get_emg_data(self):
        with self.lock:
            print("H")

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.emg))
            # data = np.abs(list(event.emg))

            data = list(event.emg)
            if len(self.data) < 40:
                self.data.append(data)
            else:
                data = np.abs(self.data).mean(axis=0)
                print(predict(self.clf, data))
                self.data = []

                # if len(list(self.emg_data_queue)) >= self.n:
                #     self.emg_data_queue.clear()
                #     return False


def data_process(df, gesture):
    """
    数据处理
    :return:
    """
    data = df[[str(i) for i in range(8)]]
    data = np.abs(data)
    result_df = pd.DataFrame()
    result_data = []
    for i in range(len(data) - 40):
        # result_df = result_df.append(pd.DataFrame(data[i:(i + 40)].mean(axis=0)))
        result_data.append(data[i:(i+40)].mean(axis=0).tolist())
    result_df = pd.DataFrame(result_data)
    result_df['gesture'] = gesture
    return result_df


def dump_model():
    """
    保存模型
    :return:
    """
    # df_data = pd.read_csv('output/gesture_data.csv')
    df_data = pd.read_sql('select * from genture_data', engine)
    df = pd.DataFrame()
    for gesture in df_data.gesture.unique():
        df_sub = df_data[df_data.gesture == gesture]
        df = df.append(data_process(df_sub, gesture))
    # data, labels = data_process1(df)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df['gesture'])
    # cols = [str(i) for i in range(8)]
    data = df[list(range(8))].values

    clf = RandomForestClassifier()
    print(cross_val_score(clf, data, labels, cv=5))
    clf = RandomForestClassifier()
    clf.fit(data, labels)

    # print(cross_val_score(clf, data, labels, cv=5))
    # pickle.dumps('models/randomforest.pkl', clf)
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(clf, f)

    with open('models/le.pkl', 'wb') as f:
        pickle.dump(le, f)


def load_model():
    """
    加载模型
    :return:
    """
    f = open('models/random_forest.pkl', 'rb')
    clf = pickle.load(f)

    myo.init(sdk_path='sdk')
    # while True:
    #     df = collect(myo, 'unknown')
    #     data = df[list(range(8))].values
    #     labels = clf.predict(data)
    #
    #     # for y in labels:
    #     print()
    while True:
        hub = myo.Hub()
        listener = Listener(2000, clf)
        hub.run(listener.on_event, 20000)


def predict(clf, data):
    f = open('models/le.pkl', 'rb')
    le = pickle.load(f)
    return le.inverse_transform(clf.predict([data]))


if __name__ == '__main__':
    # load_model()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('f', choices=['train', 'load'], help='func name')
    args = parser.parse_args()
    if args.f == 'train':
        dump_model()
    if args.f == 'load':
        load_model()
