import pickle
from autogluon.timeseries.learner import TimeSeriesLearner
from autogluon.timeseries.predictor import TimeSeriesPredictor
from autogluon.timeseries.evaluator import TimeSeriesEvaluator
# from autogluon.timeseries.utils.forecast import
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

root = 'AutogluonModels/ag-20230317_153718'

# learner = pickle.load(open(f'{root}/learner.pkl', mode='rb'))
# print(type(learner))

# print(learner.get_info())
predictor = TimeSeriesPredictor()
predictor.load(r'L:\pycharm projects\Hybrid_model\AutogluonModels\ag-20230319_043134')


print(type(predictor))
# print(f'model_names: {predictor.get_model_names()}|best_model: {predictor.get_model_best()}')
print(f'target: {predictor.target}')

test_data = TimeSeriesDataFrame('resample/resample/16-21merged_logs_interpolated_resample_20_min_interval.csv')
