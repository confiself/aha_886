#! coding:utf-8

import re
import numpy as np
import zipfile
import os
import predict_helper
import pandas as pd
import copy
import json
import matplotlib.pyplot as plt

DATA_FILE_NAME = 'traffic_flow_prediction_data.zip'


def _parse_data(file_name, mode='train'):
    data = pd.read_csv(file_name, names=['date', 'cross_name', 'location', 'direct_1_value', 'direct_2_value'])
    data['date_parse'] = data['date'].map(predict_helper.get_date_info)
    data['cross_name'] = data['cross_name'].map(predict_helper.get_cross_name)
    data['location'] = data['location'].map(predict_helper.get_location_name)
    hcd = data['date_parse'].map(lambda x: x[2])
    if mode != 'train':
        hcd = data['date'].map(lambda x: x.split()[0])
    base_data = pd.DataFrame({'weather': 1,
                              'hcd': hcd,
                              'minute': data['date_parse'].map(lambda x: x[0]),
                              'weekday': data['date_parse'].map(lambda x: x[1]),
                              'cross_name': data['cross_name'],
                              'location': data['location']
                              })
    base_data_direct_1 = copy.deepcopy(base_data)
    base_data_direct_2 = copy.deepcopy(base_data)
    base_data_direct_1['direct'] = 0
    base_data_direct_1['value'] = data['direct_1_value']
    base_data_direct_2['direct'] = 1
    base_data_direct_2['value'] = data['direct_2_value']
    return pd.concat([base_data_direct_1, base_data_direct_2])


def _pre_process_data():
    _data_dir = 'data'
    valid_date_path = '01-17'
    if not os.path.exists(valid_date_path):
        f = zipfile.ZipFile('data/' + DATA_FILE_NAME, 'r')
        for _file in f.namelist():
            f.extract(_file, "data/")
    columns = ['weather', 'hcd', 'minute', 'weekday', 'cross_name', 'location', 'direct', 'value']
    train_data = pd.DataFrame(columns=columns)
    valid_data = pd.DataFrame(columns=columns)
    for _dates in os.listdir(_data_dir):
        file_dates_path = os.path.join(_data_dir, _dates)
        if not re.match(r'\d{2}-\d{2}', _dates):
            continue
        for file_name in os.listdir(file_dates_path):
            file_path = os.path.join(file_dates_path, file_name)
            _data = _parse_data(file_path, mode='train')
            if _dates == valid_date_path:
                valid_data = valid_data.append(_data)
            else:
                train_data = train_data.append(_data)

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_data.to_csv('model/train_data', index=False, float_format='%.0f', columns=columns)
    valid_data.to_csv('model/valid_data', index=False, float_format='%.0f', columns=columns)


def _pre_process_data_gen():
    _data_dir = 'data'
    valid_date_path = '01-17'
    if not os.path.exists(valid_date_path):
        f = zipfile.ZipFile('data/' + DATA_FILE_NAME, 'r')
        for _file in f.namelist():
            f.extract(_file, "data/")
    columns = ['weather', 'hcd', 'minute', 'weekday', 'cross_name', 'location', 'direct', 'value']
    train_data = pd.DataFrame(columns=columns)
    valid_data = pd.DataFrame(columns=columns)
    for _dates in os.listdir(_data_dir):
        file_dates_path = os.path.join(_data_dir, _dates)
        if not re.match(r'\d{2}-\d{2}', _dates):
            continue
        for file_name in os.listdir(file_dates_path):
            file_path = os.path.join(file_dates_path, file_name)
            _data = _parse_data(file_path, mode='gen')
            if _dates == valid_date_path:
                valid_data = valid_data.append(_data)
            train_data = train_data.append(_data)

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_data.to_csv('model/train_data_d', index=False, float_format='%.0f', columns=columns)
    valid_data.to_csv('model/valid_data_d', index=False, float_format='%.0f', columns=columns)


def get_train_data():
    data = pd.read_csv('model/train_data')
    data = data.groupby(by=['weather', 'hcd', 'minute', 'cross_name', 'weekday']).agg({'value': sum}).reset_index()
    y_data = data['value'].values
    x_data = data.drop('value', axis=1).values
    x_data = x_data[:, [0, 1, 2, 4, 3]]
    x_data = np.divide(x_data, np.array(predict_helper.normalize_params(), dtype=float))
    train_size = int(0.9 * x_data.shape[0])
    x_train, y_train, x_test, y_test = x_data[0: train_size], y_data[0:train_size], \
                                       x_data[train_size:], y_data[train_size:]
    return x_train, y_train, x_test, y_test


def get_seq_train_data():
    data = pd.read_csv('model/train_data')
    data = data.groupby(by=['weather', 'hcd', 'minute', 'cross_name', 'weekday']).agg({'value': sum}).reset_index()
    y_data = data['value'].values
    x_data = data.drop('value', axis=1).values
    # x_data = data.values
    x_data = x_data[:, [0, 1, 2, 4, 3]]
    x_data_seq = []
    y_data_seq = []
    for i, _data in enumerate(x_data):
        if i < 12:
            continue
        x_data_seq.append(x_data[i - 12: i])
        y_data_seq.append(y_data[i])
    x_data = np.array(x_data_seq)
    y_data = np.array(y_data_seq)
    x_data = np.divide(x_data, np.array(predict_helper.normalize_params(), dtype=float))
    return x_data, y_data


class RawData(object):
    def __init__(self):
        data = pd.read_csv('model/train_data_d')
        data = data.groupby(by=['weather', 'hcd', 'minute', 'cross_name', 'weekday']).agg({'value': sum}).reset_index()
        self.data = data.values
        self._cross_names = ('wuhe_zhangheng', 'wuhe_jiaxian', 'wuhe_longping',
                             'chongzhi_jiaxian', 'chongzhi_longping', 'chongzhi_beier')

    def get_data(self, date_str, cross_name):
        cross_name = predict_helper.ROAD_CROSS_NAMES[cross_name]
        return filter(lambda x: x[1] == date_str and x[3] == cross_name, self.data)

    def plot_data(self, date_str, cross_name):
        data = self.get_data(date_str, cross_name)
        plt.plot([x[2] for x in data], [x[-1] for x in data])
        plt.show()

    def plot_data_list(self, date_str_list, cross_name):
        colors = ['r', 'g', 'b']
        data_before = None
        for i, date_str in enumerate(date_str_list):
            data = self.get_data(date_str, cross_name)
            if len(data) != 288:
                continue
            if data_before:
                total = sum([x[-1] for x in data]) - sum([x[-1] for x in data_before])
                print total / 288, 'diff', date_str
            data_before = data
            print len(data), date_str

            plt.plot([x[2] for x in data], [x[-1] for x in data], color=colors[i])

        plt.show()

    @staticmethod
    def write_data_debug():
        raw_data = RawData()
        cross_dict = {'chongzhi_beier': ['2019/1/28', '2019/1/31'],
                      'chongzhi_jiaxian': ['2019/1/28', '2019/1/31'],
                      'chongzhi_longping': ['2019/1/28', '2019/1/31'],
                      'wuhe_jiaxian': ['2019/1/28', '2019/1/31'],
                      'wuhe_longping': ['2019/1/28', '2019/1/31'],
                      'wuhe_zhangheng': ['2019/1/28', '2019/1/31'],
                      }
        resp_data = {}
        for cross_name, date_str_list in cross_dict.items():
            _data = []
            for data_str in date_str_list:
                _data += raw_data.get_data(data_str, cross_name)
            resp_data[cross_name] = [x[-1] for x in _data]
            assert len(_data) == 288 * 2
        with open('model/debug.txt', 'w') as f_w:
            f_w.writelines(json.dumps(resp_data))

    @staticmethod
    def write_data_product(mode='1'):
        raw_data = RawData()
        if mode == '1':
            cross_dict = {'chongzhi_beier': ['2019/1/28', '2019/1/17'],
                          'chongzhi_jiaxian': ['2019/1/28', '2019/1/17'],
                          'chongzhi_longping': ['2019/1/28', '2019/1/17'],
                          'wuhe_jiaxian': ['2019/1/28', '2019/1/17'],
                          'wuhe_longping': ['2019/1/28', '2019/1/17'],
                          'wuhe_zhangheng': ['2019/1/28', '2019/1/17'],
                          }
        elif mode == '2':
            cross_dict = {'chongzhi_beier': ['2019/1/28', '2019/1/24'],
                          'chongzhi_jiaxian': ['2019/1/28', '2019/1/24'],
                          'chongzhi_longping': ['2019/1/28', '2019/1/24'],
                          'wuhe_jiaxian': ['2019/1/28', '2019/1/24'],
                          'wuhe_longping': ['2019/1/28', '2019/1/24'],
                          'wuhe_zhangheng': ['2019/1/28', '2019/1/24'],
                          }
        else:
            cross_dict = {'chongzhi_beier': ['2019/1/28', '2019/1/17'],
                          'chongzhi_jiaxian': ['2019/1/28', '2019/1/17'],
                          'chongzhi_longping': ['2019/1/28', '2019/1/17'],
                          'wuhe_jiaxian': ['2019/1/28', '2019/1/17'],
                          'wuhe_longping': ['2019/1/28', '2019/1/17'],
                          'wuhe_zhangheng': ['2019/1/28', '2019/1/17'],
                          }
        resp_data = {}
        for cross_name, date_str_list in cross_dict.items():
            _data = []
            for data_str in date_str_list:
                cur_data = raw_data.get_data(data_str, cross_name)
                cur_data = [x[-1] for x in cur_data]
                if mode == '3' and data_str == '2019/1/17':
                    for i in range(144, 168):
                        cur_data[i] += 3
                    for i in range(216, len(cur_data)):
                        cur_data[i] += 3
                _data += cur_data
            resp_data[cross_name] = _data
            assert len(_data) == 288 * 2
        with open('model/product_{}.txt'.format(mode), 'w') as f_w:
            f_w.writelines(json.dumps(resp_data))

    def plot_data_product(self):
        data_list = []
        colors = ['r', 'g', 'b']
        for i in [1, 2, 3]:
            with open('model/product_{}.txt'.format(i)) as f:
                line = f.readline()
                w = eval(line)
                _data = []
                for cross_name in self._cross_names:
                    _data += w[cross_name]
                data_list.append(_data)
        for i, _data in enumerate(data_list):
            if i == 1:
                continue
            plt.plot(_data[-288:], color=colors[i])
        plt.show()
        diff = (sum(data_list[2]) - sum(data_list[0]))
        print(diff)

    @staticmethod
    def self_test():
        raw_data = RawData()
        raw_data.plot_data_list(['2019/1/21', '2019/1/28'], cross_name='chongzhi_beier')
        raw_data.plot_data_list(['2019/1/17', '2019/1/24', '2019/1/31'], cross_name='chongzhi_beier')
        raw_data.write_data_product('1')

if __name__ == '__main__':
    _pre_process_data()
    # _pre_process_data_gen()
    # raw_data = RawData()
    # raw_data.plot_data_product()