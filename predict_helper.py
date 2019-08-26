#! coding:utf-8
import datetime
import numpy as np
START_MINUTE = 5 * 60
STOP_MINUTE = 21 * 60
import datetime
ROAD_CROSS_NAMES = {'chongzhi_beier': 0, 'chongzhi_jiaxian': 1, 'chongzhi_longping': 2,
                    'wuhe_jiaxian': 3, 'wuhe_longping': 4, 'wuhe_zhangheng': 5}
ROAD_LOCATION_NAMES = {'east': 0, 'south': 1, 'west': 2, 'north': 3}

DIRECTIONS = (0, 1)

# 天气，离放假前的天数，分钟段，周几 ,路口
NORMALIZE_PARAMS = [1, 30, 24 * 60, 7, 5, 500]


def normalize_params(mode='dense'):
    if mode == 'dense':
        return NORMALIZE_PARAMS[:-1]
    return NORMALIZE_PARAMS


def get_cross_name(x):
    return ROAD_CROSS_NAMES[x]


def get_location_name(x):
    return ROAD_LOCATION_NAMES[x]


def get_date_info(date_str):
    _time_format = '%Y/%m/%d %H:%M:%S'
    if ':' not in date_str:
        _time_format = '%Y/%m/%d'
    record_date = datetime.datetime.strptime(date_str, _time_format)
    minute = record_date.minute + record_date.hour * 60
    week_day = record_date.weekday().real
    holiday_count_down = abs(datetime.datetime(2019, 2, 4, 23, 59, 59) - record_date).days
    return minute, week_day, holiday_count_down


def get_predict_data(date_str, cross_name, match_level):
    _, week_day, holiday_count_down = get_date_info(date_str)
    if date_str == '2019/02/07':
        week_day = 6
    cross_name = ROAD_CROSS_NAMES[cross_name]
    x_predict = []
    for minute in range(0, 24 * 60, 5):
        x_predict.append([1, holiday_count_down, minute, week_day, cross_name])
    if match_level == 'heat':
        x_predict = filter(lambda _x: START_MINUTE <= _x[2] < STOP_MINUTE, x_predict)
    x_predict = np.array(x_predict) / np.array(normalize_params(), dtype=float)
    x_predict = x_predict.tolist()
    return x_predict


def get_predict_seq_data(date_str, cross_name):
    _, week_day, holiday_count_down = get_date_info(date_str)
    _predict_time = datetime.datetime.strptime(date_str, '%Y/%m/%d')
    _last_time = _predict_time - datetime.timedelta(days=1)
    _last_time_str = _last_time.strftime('%Y/%m/%d')
    _, week_day_before, holiday_count_down_before = get_date_info(_last_time_str)
    if date_str == '2019/02/07':
        week_day = 6
        week_day_before = 5
    cross_name = ROAD_CROSS_NAMES[cross_name]
    x_predict = [[1, holiday_count_down_before, minute, week_day_before, cross_name]
                 for minute in range(0, 24 * 60, 5)]
    x_predict += [[1, holiday_count_down, minute, week_day, cross_name]
                  for minute in range(0, 24 * 60, 5)]
    x_predict_data = []
    for i in range(len(x_predict)/2, len(x_predict)):
        x_predict_data.append(x_predict[i-12: i])
    return x_predict_data
