from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from data_prepos import op_dir, resamp

resamp('22-29-01-16merged_interpolated', 20)
interpolated_data = TimeSeriesDataFrame(
    f'{op_dir}/resample/22-29-01-16merged_interpolated_resample_20_min_interval.csv')

predictor = TimeSeriesPredictor(target='DE_AREA', prediction_length=100, prests='best_quality')
predictor.fit(interpolated_data, error='raise')

# prediction = learner.predict(interpolated_data)

leader_board = predictor.leaderboard(interpolated_data)
print(leader_board)
