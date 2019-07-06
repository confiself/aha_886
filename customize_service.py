import json
import data_helper

from model_service.tfserving_model_service import TfServingBaseService


class PredictService(TfServingBaseService):

    def _preprocess(self, data):
        match_level = 'heat'
        mode = 'debug'
        if mode == 'debug':
            _dates = ('2019/02/04', '2019/02/07')
        else:
            _dates = ('2019/02/11', '2019/02/14')

        if match_level == 'heat':
            _cross_names = ('wuhe_zhangheng',)
        else:
            _cross_names = ('wuhe_zhangheng', 'wuhe_jiaxian', 'wuhe_longping',
                            'chongzhi_jiaxian', 'chongzhi_longping', 'chongzhi_beier')

        data = []
        for _cross_name in _cross_names:
            for _date in _dates:
                data += data_helper.get_predict_data(_date, _cross_name, match_level)
        print("end to pre process total {}".format(len(data)))
        return {"inputs": data}

    def _postprocess(self, data):
        print("begin to post process")
        data = data_helper.parse_resp_data(data['result'])
        schema = json.dumps({'wuhe_zhangheng': data})
        print("end to post process")
        return {"data": {"resp_data": schema}}
