#! coding:utf-8

from sklearn.preprocessing import StandardScaler
import datetime
import os
import re
import numpy as np
import zipfile
ROAD_CROSS_NAMES = {'chongzhi_beier': 0, 'chongzhi_jiaxian': 1, 'chongzhi_longping': 2,
                    'wuhe_jiaxian': 3, 'wuhe_longping': 4, 'wuhe_zhangheng': 5}

DATA_FILE_NAME = 'traffic_flow_prediction_data.zip'


def _parse_file(file_path, data):
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            line = line.split(',')
            assert len(line) == 5
            date_key = ','.join(line[:2])
            flow_value = int(line[-1]) + int(line[-2])
            if date_key in data:
                data[date_key] += flow_value
            else:
                data[date_key] = flow_value
    return data


def _parse_data(date_key):
    date_key_list = date_key.split(',')
    date_str = date_key_list[0]
    cross_name = date_key_list[1]
    minute, week_day, holiday_count_down = get_date_info(date_str)
    return {'minute': minute, 'week_day': week_day,
            'holiday_count_down': holiday_count_down, 'cross_name': ROAD_CROSS_NAMES[cross_name]}


def get_date_info(date_str):
    _time_format = '%Y/%m/%d %H:%M:%S'
    if ':' not in date_str:
        _time_format = '%Y/%m/%d'
    record_date = datetime.datetime.strptime(date_str, _time_format)
    minute = record_date.minute + record_date.hour * 60
    week_day = record_date.weekday().real
    holiday_count_down = abs(datetime.datetime(2019, 2, 4, 23, 59, 59) - record_date).days
    return minute, week_day, holiday_count_down
    

def _pre_process_data():
    _data_dir = 'data'
    if not os.path.exists('data/01-12'):
        f = zipfile.ZipFile('data/' + DATA_FILE_NAME, 'r')
        for _file in f.namelist():
            f.extract(_file, "data/")
    data = {}
    for _dates in os.listdir(_data_dir):
        file_dates_path = os.path.join(_data_dir, _dates)
        if not re.match(r'\d{2}-\d{2}', _dates):
            continue
        for file_name in os.listdir(file_dates_path):
            file_path = os.path.join(file_dates_path, file_name)
            _parse_file(file_path, data)

    data_list = sorted(data.items(), key=lambda x: x[0])
    print 'data len: {}, data[0]: {}'.format(len(data_list), data_list[0])
    train_data = []
    valid_data = []
    for (date_key, value) in data_list:
        record = _parse_data(date_key)
        # 天气，离放假前的天数，分钟段，周几 ,路口, 路口总流量
        data_item = [1, record['holiday_count_down'], record['minute'],
                     record['week_day'], record['cross_name'], value]
        if date_key.startswith('2019/1/17'):
            valid_data.append(data_item)
        else:
            train_data.append(data_item)
    # 数据归一化
    scaller = StandardScaler()
    train_data = scaller.fit_transform(train_data)
    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np.savetxt('model/scaller.model', [scaller.mean_, scaller.scale_])
    np.savetxt('model/valid_data', valid_data, fmt='%d')
    np.savetxt('model/train_data', train_data)


def get_train_data():
    data = np.loadtxt('model/train_data')
    y_data = data[:, -1]
    x_data = np.delete(data, -1, axis=1)
    train_size = int(0.9 * data.shape[0])
    x_train, y_train, x_test, y_test = x_data[0: train_size], y_data[0:train_size], \
                                       x_data[train_size:], y_data[train_size:]
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    _pre_process_data()
