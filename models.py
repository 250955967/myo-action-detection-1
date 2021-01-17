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

from collect_data import data_process1, collect


def dump_model():
    """
    保存模型
    :return:
    """
    df = pd.read_csv('output/gesture_data.csv')
    # data, labels = data_process1(df)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df['gesture'])
    cols = [str(i) for i in range(8)]
    data = df[cols].values

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
    f = open('models/le.pkl', 'rb')
    le = pickle.load(f)
    myo.init(sdk_path='sdk')
    while True:
        df = collect(myo, 'unknown')
        data = df[list(range(8))].values
        labels = clf.predict(data)

        # for y in labels:
        print(le.inverse_transform(labels))


if __name__ == '__main__':
    load_model()
