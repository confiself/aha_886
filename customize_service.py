import json
import predict_helper

from model_service.tfserving_model_service import TfServingBaseService


class PredictService(TfServingBaseService):

    def _preprocess(self, data):
        self._match_level = 'final'
        self._mode = 'product'
        if self._mode == 'debug':
            self._dates = ('2019/02/04', '2019/02/07')
        else:
            self._dates = ('2019/02/11', '2019/02/14')

        if self._match_level == 'heat':
            self._cross_names = ('wuhe_zhangheng',)
        else:
            self._cross_names = ('wuhe_zhangheng', 'wuhe_jiaxian', 'wuhe_longping',
                            'chongzhi_jiaxian', 'chongzhi_longping', 'chongzhi_beier')

        data = []
        for _cross_name in self._cross_names:
            for _date in self._dates:
                data += predict_helper.get_predict_data(_date, _cross_name, self._match_level)
        print("end to pre process total {}".format(len(data)))
        return {"inputs": data}

    def _postprocess(self, data):
        print("begin to post process")
        data = [x[0] if isinstance(x, list) else x for x in data['result']]
        cross_num = len(self._cross_names)
        data_num_per_day = 576 if self._match_level == 'final' else 384
        resp_data = {}
        for i in range(cross_num):
            resp_data[self._cross_names[i]] = data[i * data_num_per_day: (i+1) * data_num_per_day]
        print("end to post process")
        return {"data": {"resp_data": json.dumps(resp_data)}}
