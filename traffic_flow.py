#! coding:utf-8
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
import data_helper
import numpy as np
import copy

# import matplotlib.pyplot as plt

BATCH_SIZE = 16
NUM_EPOCHS = 10
INPUT_DIM = 5
START_MINUTE = 5 * 60
STOP_MINUTE = 21 * 60


def train():
    model = Sequential()
    model.add(Dense(768, input_dim=INPUT_DIM, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='glorot_uniform'))
    model.compile(loss='mse', optimizer='adam')
    x_train, y_train, x_test, y_test = data_helper.get_train_data()
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)
    model.save('model/model.h5')
    # plot loss
    # plt.plot(history.history['loss', 'val_loss'])
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def predict_custom_date(date_str, cross_name, weather=1):
    _, week_day, holiday_count_down = data_helper.get_date_info(date_str)
    return _predict(holiday_count_down, week_day, cross_name, weather)


def _predict(holiday_count_down, week_day, cross_name, weather=1):
    scale_params = np.loadtxt('model/scaller.model')

    def _cross_name(x):
        if isinstance(x, str):
            return data_helper.ROAD_CROSS_NAMES[x]
        return x

    def _fit(data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = (data[i][j] - scale_params[0][j]) / scale_params[1][j]
        return data

    x_predict = [[weather, holiday_count_down, minute, week_day, _cross_name(cross_name)]
                 for minute in range(0, 24 * 60, 5)]
    model = load_model('model/model.h5')
    x_predict_fit = _fit(copy.deepcopy(x_predict))
    out = model.predict([x_predict_fit]).flatten()
    return [{'minute': x_predict[index][2], 'value': int(out[index] * scale_params[1][-1] + scale_params[0][-1])}
            for index in range(len(x_predict))]


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
        hour_weights[key] /= total_flow
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


def submit(mode='debug', match_level='heat'):
    """
    :param mode: debug/product
    :param match_level: heat/final
    :return:
    """
    if mode == 'debug':
        _dates = ('2019/02/04', '2019/02/07')
    else:
        _dates = ('2019/02/11', '2019/02/14')

    if match_level == 'heat':
        _cross_names = ('wuhe_zhangheng',)
    else:
        _cross_names = ('wuhe_zhangheng', 'wuhe_jiaxian', 'wuhe_longping',
                        'chongzhi_jiaxian', 'chongzhi_longping', 'chongzhi_beier')

    resp_data = {}
    for cross_name in _cross_names:
        values = []
        for date in _dates:
            result = predict_custom_date(date, cross_name)
            if match_level == 'heat':
                result = filter(lambda _x: START_MINUTE <= _x['minute'] < STOP_MINUTE, result)
            values += [x['value'] for x in result]
        resp_data[cross_name] = values
    return {'data': {'resp_data': resp_data}}


def evaluate():
    cross_name = 'wuhe_zhangheng'
    data_predict = predict_custom_date('2019/1/17', cross_name)
    valid_data = np.loadtxt('model/valid_data')
    data_true = [{'minute': x[2], 'value': x[5]}
                 for x in valid_data if x[4] == data_helper.ROAD_CROSS_NAMES[cross_name]]
    data_true = filter(lambda _x: START_MINUTE <= _x['minute'] < STOP_MINUTE, data_true)
    data_predict = filter(lambda _x: START_MINUTE <= _x['minute'] < STOP_MINUTE, data_predict)
    score = _get_classify_score(data_predict, data_true) * 0.4 + \
            _get_regression_score(data_predict, data_true) * 0.6
    print('score: {}'.format(score))


if __name__ == '__main__':
    train()
    evaluate()
    # submit()

