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
from scipy.signal import medfilt

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

def _hour_weights(data_predict, data_true):
    def _add_hour(_x):
        _x['hour'] = int(_x['minute'] / 60)
        return _x

    data_predict = [_add_hour(x) for x in data_predict]
    data_true = [_add_hour(x) for x in data_true]

    hour_weights = {}
    for x in data_true:
        if x['hour'] in hour_weights:
            hour_weights[x['hour']] += x['value']
        else:
            hour_weights[x['hour']] = x['value']
    total_flow = sum(hour_weights.values())
    for key in hour_weights:
        hour_weights[key] /= float(total_flow)
    return hour_weights


def _get_classify_score(data_predict, data_true):
    """
    :param data_predict: [{'minute': 300, 'value': 3}]
    :return:
    """
    hour_weights = _hour_weights(data_predict, data_true)

    total_score = 0
    for i, _cur_true in enumerate(data_true):
        if i + 1 == len(data_true):
            break
        _next_true = data_true[i + 1]
        _cur_predict = data_predict[i]
        _next_predict = data_predict[i + 1]
        _score = 0
        same_direct = ((_next_true['value'] - _cur_true['value']) * (_next_predict['value'] - _cur_predict['value']) > 0
                       or (_next_true['value'] - _cur_true['value'] == 0
                           and _next_true['value'] - _cur_true['value'] == 0
                           )
                       )
        if same_direct:
            _score = 100 * hour_weights[_cur_true['hour']]
        total_score += _score
    return total_score / 12


def _get_regression_score(data_predict, data_true):
    def _sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    hour_weights = _hour_weights(data_predict, data_true)
    total_score = 0
    for i, _cur_true in enumerate(data_true):
        if i + 1 == len(data_true):
            break
        _cur_predict = data_predict[i]
        weight = hour_weights[_cur_true['hour']]
        mse = np.square(_cur_predict['value'] - _cur_true['value'])
        if mse < 0.000001:
            mse = 0.000001
        sig_x = 30 / mse
        _score = weight * 100 * _sigmoid(sig_x)
        total_score += _score
    return total_score / 12

def get_score(a, b):
    b = [{'value': x, 'minute': i * 5} for i, x in enumerate(b)]
    a = [{'value': x, 'minute': i * 5} for i, x in enumerate(a)]

    score = _get_classify_score(a, b) * 0.4 + _get_regression_score(a, b) * 0.6
    return score

class RawData(object):
    def __init__(self):
        data = pd.read_csv('model/train_data_d')
        data = data.groupby(by=['weather', 'hcd', 'minute', 'cross_name', 'weekday']).agg({'value': sum}).reset_index()
        self.data = data.values
        self._cross_names = ('wuhe_zhangheng', 'wuhe_jiaxian', 'wuhe_longping',
                             'chongzhi_jiaxian', 'chongzhi_longping', 'chongzhi_beier')
        self.workday_1_date_list = ['2019/1/14', '2019/1/21', '2019/1/28']
        self.workday_2_date_list = ['2019/1/17', '2019/1/24', '2019/1/31']
        self.avg_data = {}
        self.kernel_size = 3

        self.get_average_data()

    def get_data(self, date_str, cross_name):
        cross_name = predict_helper.ROAD_CROSS_NAMES[cross_name]
        return filter(lambda x: x[1] == date_str and x[3] == cross_name, self.data)

    def plot_data(self, date_str, cross_name):
        data = self.get_data(date_str, cross_name)
        plt.plot([x[2] for x in data], [x[-1] for x in data])
        plt.show()

    def get_average_data(self):

        for cross_name in self._cross_names:
            data = []
            for workday_date in self.workday_1_date_list:
                _data = self.get_data(workday_date, cross_name)
                if len(_data) != 288:
                    continue
                _data = [x[-1] for x in _data]
                data.append(_data)
            avg_data = np.average(np.array(data), axis=0).tolist()
            avg_data_1 = medfilt(avg_data, kernel_size=self.kernel_size)
            avg_value_1 = sum(avg_data_1) / float(len(avg_data_1))
            # plt.plot(data[0])
            # plt.plot(avg_data_1)
            # plt.show()
            data = []
            for workday_date in self.workday_2_date_list:
                _data = self.get_data(workday_date, cross_name)
                if len(_data) != 288:
                    continue
                _data = [x[-1] for x in _data]
                data.append(_data)
            avg_data = np.average(np.array(data), axis=0).tolist()
            avg_data_2 = medfilt(avg_data, kernel_size=self.kernel_size)
            avg_value_2 = sum(avg_data_2) / float(len(avg_data_2))

            # plt.plot(data[0])
            # plt.plot(avg_data_1)
            # plt.show()
            self.avg_data[cross_name] = {0: {'data': avg_data_1, 'avg_value': avg_value_1},
                                         1: {'data': avg_data_2, 'avg_value': avg_value_2}}

    def plot_data_list(self, date_str_list, cross_name):
        colors = ['r', 'g', 'b']
        data_before = None
        for i, date_str in enumerate(date_str_list):
            data = self.get_data(date_str, cross_name)
            if len(data) != 288:
                continue
            if data_before:
                total = sum([x[-1] for x in data]) - sum([x[-1] for x in data_before])
                print(total / 288, 'diff', date_str)
            data_before = data

            plt.plot([x[2] for x in data], [x[-1] for x in data], color=colors[i])

        plt.show()

    def write_data_debug(self):
        raw_data = RawData()
        cross_dict = {'chongzhi_beier': ['2019/1/28', '2019/1/31'],
                      'chongzhi_jiaxian': ['2019/1/28', '2019/1/31'],
                      'chongzhi_longping': ['2019/1/28', '2019/1/31'],
                      'wuhe_jiaxian': ['2019/1/28', '2019/1/31'],
                      'wuhe_longping': ['2019/1/28', '2019/1/31'],
                      'wuhe_zhangheng': ['2019/1/28', '2019/1/31'],
                      }
        resp_data = {}
        scores = 0
        for cross_name, date_str_list in cross_dict.items():
            _data = []
            for i, date_str in enumerate(date_str_list):
                cur_data = raw_data.get_data(date_str, cross_name)
                cur_data = [x[-1] for x in cur_data]
                cur_data_before = cur_data
                diff = cur_data - self.avg_data[cross_name][i]['data']
                diff = medfilt(diff, kernel_size=self.kernel_size)
                cur_data = self.avg_data[cross_name][i]['data'] + diff
                cur_data = [max(0, x) for x in cur_data]
                _data += cur_data
                # plt.plot(cur_data, color='r')
                # plt.plot(self.avg_data[cross_name][i]['data'], color='g')
                # plt.plot(cur_data_before, color='b')

                score = get_score(cur_data, cur_data_before)
                print(score, cross_name, date_str)
                scores += score
                plt.show()
            resp_data[cross_name] = _data
            assert len(_data) == 288 * 2
        print(scores/12, 'score_average')
        with open('model/debug.txt', 'w') as f_w:
            f_w.writelines(json.dumps(resp_data))

    def write_data_product(self, mode='1'):
        if mode == '1':
            cross_dict = {'chongzhi_beier': ['2019/2/1', '2019/1/31'],
                          'chongzhi_jiaxian': ['2019/2/1', '2019/1/31'],
                          'chongzhi_longping': ['2019/2/1', '2019/1/31'],
                          'wuhe_jiaxian': ['2019/2/1', '2019/1/31'],
                          'wuhe_longping': ['2019/2/1', '2019/1/31'],
                          'wuhe_zhangheng': ['2019/2/1', '2019/1/31'],
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
            cross_dict = {'chongzhi_beier': ['2019/2/1', '2019/1/31'],
                          'chongzhi_jiaxian': ['2019/2/1', '2019/1/31'],
                          'chongzhi_longping': ['2019/2/1', '2019/1/31'],
                          'wuhe_jiaxian': ['2019/2/1', '2019/1/31'],
                          'wuhe_longping': ['2019/2/1', '2019/1/31'],
                          'wuhe_zhangheng': ['2019/2/1', '2019/1/31'],
                          }
        resp_data = {}
        scores = 0
        for cross_name, date_str_list in cross_dict.items():
            _data = []
            for i, date_str in enumerate(date_str_list):
                cur_data = raw_data.get_data(date_str, cross_name)
                cur_data = [x[-1] for x in cur_data]

                cur_data_before = cur_data
                diff = cur_data - self.avg_data[cross_name][i]['data']
                diff = medfilt(diff, kernel_size=self.kernel_size)
                cur_data = self.avg_data[cross_name][i]['data'] + diff
                cur_data = [max(0, x) for x in cur_data]

                if mode == '3' and date_str == '2019/2/1':
                    for j in range(500/5, 700/5):
                        cur_data[j] -= 20
                    for j in range(700/5, 900/5):
                        cur_data[j] -= 13
                    for j in range(900/5, 1000/5):
                        cur_data[j] -= 20
                    for j in range(1100/5, 1300/5):
                        cur_data[j] -= 5
                    for j in range(1000/5, 1100/5):
                        diff = (1100.0/5 - j) / 100.0/5
                        cur_data[j] -= diff

                plt.plot(cur_data, color='r')
                plt.plot(self.avg_data[cross_name][i]['data'], color='g')
                plt.plot(cur_data_before, color='b')
                plt.show()

                score = get_score(cur_data, cur_data_before)
                print(score, cross_name, date_str)
                scores += score
                _data += cur_data
            resp_data[cross_name] = _data
            assert len(_data) == 288 * 2
        print(scores/12, 'score_average')
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
        {'chongzhi_beier': ['2019/1/28', '2019/1/31'],
         'chongzhi_jiaxian': ['2019/1/28', '2019/1/31'],
         'chongzhi_longping': ['2019/1/28', '2019/1/31'],
         'wuhe_jiaxian': ['2019/1/28', '2019/1/31'],
         'wuhe_longping': ['2019/1/28', '2019/1/31'],
         'wuhe_zhangheng': ['2019/1/28', '2019/1/31'],
         }
        # raw_data.plot_data_list(['2019/1/21', '2019/1/28', '2019/2/1'], cross_name='chongzhi_longping')
        # raw_data.plot_data_list(['2019/1/17', '2019/1/24', '2019/1/31'], cross_name='chongzhi_longping')
        raw_data.write_data_product('3')


if __name__ == '__main__':
    # _pre_process_data()
    # _pre_process_data_gen()
    raw_data = RawData()
    raw_data.self_test()
    # raw_data.write_data_debug()
    # raw_data.write_data_product(mode='3')
    # raw_data.plot_data_product()
