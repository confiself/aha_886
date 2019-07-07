#! coding:utf-8
from keras.layers import Dense, Dropout, Input
from keras.models import load_model, Model
import data_helper
import predict_helper
import numpy as np
import shutil
import tensorflow as tf
import os
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt

BATCH_SIZE = 32
NUM_EPOCHS = 10


def train():
    features = Input(shape=(len(predict_helper.NORMALIZE_PARAMS), ))
    x = Dense(768, activation='relu', kernel_initializer='glorot_uniform')(features)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(x)
    out = Dense(1, kernel_initializer='glorot_uniform')(x)
    drop_out = Dropout(0.1)(out)
    model = Model(inputs=[features], outputs=[drop_out])
    predict_model = Model(inputs=[features], outputs=[out])

    model.compile(loss='mse', optimizer='adam')
    x_train, y_train, x_test, y_test = data_helper.get_train_data()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)
    predict_model.save('model/model.h5')
    export_model(predict_model, 'model/')


def export_model(model, export_path, export_version=1):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'inputs': model.input}, outputs={'result': model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predictions': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()


def predict_custom_date(date_str, cross_name):
    x_predict = predict_helper.get_predict_data(date_str, cross_name, 'debug')
    model = load_model('model/model.h5')
    x_predict_fit = x_predict / np.array(predict_helper.NORMALIZE_PARAMS, dtype=float)
    out = model.predict([x_predict_fit]).flatten()
    out = predict_helper.merge_location_direct(out)
    return [{'minute': x_predict[index * 8][2], 'value': out[index]}
            for index in range(len(out))]


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


def evaluate():
    cross_name = 'wuhe_zhangheng'
    data_predict = predict_custom_date('2019/02/04', cross_name)
    valid_data = pd.read_csv('model/valid_data')
    valid_data = valid_data[valid_data['cross_name'] == predict_helper.ROAD_CROSS_NAMES[cross_name]]
    valid_data = valid_data.groupby(by=['minute'])['value'].sum().tolist()
    data_true = [{'minute': x['minute'], 'value': valid_data[i]}
                 for i, x in enumerate(data_predict)]
    data_true = filter(lambda _x: predict_helper.START_MINUTE <= _x['minute'] < predict_helper.STOP_MINUTE, data_true)

    data_predict = filter(lambda _x: predict_helper.START_MINUTE <= _x['minute'] < predict_helper.STOP_MINUTE, data_predict)
    score = _get_classify_score(data_predict, data_true) * 0.4 + \
        _get_regression_score(data_predict, data_true) * 0.6
    print('score: {}'.format(score))
    plt.plot([x['minute'] for x in data_true],
             [x['value'] for x in data_true],
             color='b', label='actual')
    plt.plot([x['minute'] for x in data_predict],
             [x['value'] for x in data_predict], color='r', label='predict')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    # train()
    evaluate()
    # submit()
