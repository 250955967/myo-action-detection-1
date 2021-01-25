# -*- coding: utf-8 -*-
# CreateBy: kai
# CreateAt: 2021/1/25


import myo
from collections import deque
from threading import Lock
import time
import numpy as np
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine('mysql+pymysql://kai:password@localhost/db?charset=utf8mb4')
number_of_samples = 2000


class Listener(myo.DeviceListener):
    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.data_array = []

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
            if len(list(self.emg_data_queue)) >= self.n:
                self.data_array.append(list(self.emg_data_queue))
                self.emg_data_queue.clear()
                return False


def collect(myo, gesture):
    """
    收集数据
    :param myo:
    :return:
    """
    hub = myo.Hub()
    listener = Listener(number_of_samples)
    hub.run(listener.on_event, 20000)
    data_set = np.array((listener.data_array[0]))
    # data_set = data_process(data_set)
    df = pd.DataFrame(data_set)
    df['gesture'] = gesture
    return df


def collect_data():
    """

    :return:
    """
    # myo.init()
    myo.init(sdk_path='sdk')
    df = pd.DataFrame()
    gestures = np.loadtxt('data/gesture.csv', dtype='str')
    # gestures = gestures[:3]
    for gesture in gestures:
        print(gesture)
        input("Hold a finger movement:")
        if gesture == 'end':
            break
        data = collect(myo, gesture)
        df = df.append(data)
        # print(gesture)

    df.to_csv('output/gesture_data.csv', index=False)


if __name__ == '__main__':
    collect_data()
    # main()
