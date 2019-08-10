#! coding:utf-8

import re
import numpy as np
import zipfile
import os
import predict_helper
import pandas as pd
import copy

DATA_FILE_NAME = 'traffic_flow_prediction_data.zip'


def _parse_data(file_name):
    data = pd.read_csv(file_name, names=['date', 'cross_name', 'location', 'direct_1_value', 'direct_2_value'])
    data['date_parse'] = data['date'].map(predict_helper.get_date_info)
    data['cross_name'] = data['cross_name'].map(predict_helper.get_cross_name)
    data['location'] = data['location'].map(predict_helper.get_location_name)

    base_data = pd.DataFrame({'weather': 1,
                              'hcd': data['date_parse'].map(lambda x: x[2]),
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
            _data = _parse_data(file_path)
            if _dates == valid_date_path:
                valid_data = valid_data.append(_data)
            else:
                train_data = train_data.append(_data)

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_data.to_csv('model/train_data', index=False, float_format='%.0f', columns=columns)
    valid_data.to_csv('model/valid_data', index=False, float_format='%.0f', columns=columns)


def get_train_data():
    data = pd.read_csv('model/train_data')
    data = data.groupby(by=['weather', 'hcd', 'minute', 'cross_name', 'weekday']).agg({'value': sum}).reset_index()
    y_data = data['value'].values
    x_data = data.drop('value', axis=1).values
    x_data = x_data[:, [0, 1, 2, 4, 3]]
    x_data = np.divide(x_data, np.array(predict_helper.NORMALIZE_PARAMS, dtype=float))
    train_size = int(0.9 * x_data.shape[0])
    x_train, y_train, x_test, y_test = x_data[0: train_size], y_data[0:train_size], \
                                       x_data[train_size:], y_data[train_size:]
    return x_train, y_train, x_test, y_test


def get_seq_train_data():
    data = pd.read_csv('model/train_data')
    data = data.groupby(by=['weather', 'hcd', 'minute', 'cross_name', 'weekday']).agg({'value': sum}).reset_index()
    y_data = data['value'].values
    x_data = data.drop('value', axis=1).values
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
    x_data = np.divide(x_data, np.array(predict_helper.NORMALIZE_PARAMS, dtype=float))
    return x_data, y_data


if __name__ == '__main__':
    _pre_process_data()
