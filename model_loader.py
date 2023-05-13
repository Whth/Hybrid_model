import copy
import os

from autogluon.timeseries import *
from autogluon.timeseries.models.ensemble import TimeSeriesGreedyEnsemble

import pickle

Root_PATH = './AutogluonModels/ag-20230317_153718'


def search_model(model_dir: str, model_name: str = 'model.pkl', make_model_dict: bool = True) -> list or dict:
    """

    :param model_dir:
    :return:
    """
    model_list = []
    search_stack = [model_dir]

    while len(search_stack) > 0:
        i = search_stack.pop()
        for f in os.listdir(i):
            f_path = f'{i}/{f}'
            if os.path.isdir(f_path):
                search_stack.append(f_path)

            if os.path.basename(f) == model_name:
                model_list.append(f_path)

    print(f'at {model_dir}|found {len(model_list)}')
    model_str = ''
    for model in model_list:
        model_str += f'\t{model}\n'
    print(model_str)
    if make_model_dict:
        model_dict = {}
        for model in model_list:
            with open(model, 'rb') as f:
                temp = pickle.load(f)
            model_dict[temp.name] = [copy.deepcopy(temp), type(temp)]
        return model_dict
    return model_list


models_dict = search_model('AutogluonModels/ag-20230317_153718', make_model_dict=True)

predictions_list = []
test_data = TimeSeriesDataFrame('resample/resample/16-21merged_logs_interpolated_resample_20_min_interval.csv')


def print_model_info(model):
    print(f'{model.name}|freq: {model.freq}||is_fit: {model.is_fit()}')


ensemble_model: TimeSeriesGreedyEnsemble = models_dict.get('WeightedEnsemble')[0]
print('fit:', ensemble_model.is_fit())
print('freq:', ensemble_model.freq)
print('params:', ensemble_model.params_trained)
print(ensemble_model.model_to_weight)
print(ensemble_model.features)
print(ensemble_model.get_info())
print()
print(f'names:{ensemble_model.model_names} ')
ensemble_model.load(r'L:\pycharm projects\Hybrid_model\AutogluonModels\ag-20230317_153718\models\WeightedEnsemble/')
names = ensemble_model.model_names
data_dict = {}
for name in names:
    data_dict[name] = test_data

predictions = ensemble_model.predict(data_dict)
predictions.to_csv('prediction.csv')


print(predictions)


def method_name():
    for model_name, content in models_dict.items():
        model = content[0]
        print(f'{content[1]} predicting')
        predictions = model.predict(test_data)
        print(f'result: \n'
              f'{predictions}')
        predictions_list.append(predictions)
