#! coding:utf-8
import datetime
import numpy as np
START_MINUTE = 5 * 60
STOP_MINUTE = 21 * 60
ROAD_CROSS_NAMES = {'chongzhi_beier': 0, 'chongzhi_jiaxian': 1, 'chongzhi_longping': 2,
                    'wuhe_jiaxian': 3, 'wuhe_longping': 4, 'wuhe_zhangheng': 5}
ROAD_LOCATION_NAMES = {'east': 0, 'south': 1, 'west': 2, 'north': 3}

DIRECTIONS = (0, 1)

# 天气，离放假前的天数，分钟段，周几 ,路口，位置，方向
NORMALIZE_PARAMS = [1, 30, 24 * 60, 7, 5, 4, 1]


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
        for location in sorted(ROAD_LOCATION_NAMES.values()):
            for direct in DIRECTIONS:
                x_predict.append([1, holiday_count_down, minute, week_day, cross_name, location, direct])
    if match_level == 'heat':
        x_predict = filter(lambda _x: START_MINUTE <= _x[2] < STOP_MINUTE, x_predict)
        x_predict = np.array(x_predict) / np.array(NORMALIZE_PARAMS, dtype=float)
        x_predict = x_predict.tolist()
    return x_predict


def merge_location_direct(data):
    """前8个数值作为一组
    :param data:
    :return:
    """
    for i, value in enumerate(data):
        index = int(i / 8)
        if i != index * 8:
            data[index * 8] += value
    return [_ for x, _ in enumerate(data) if x % 8 == 0]