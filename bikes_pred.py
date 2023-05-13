from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from typing import Optional


def future_pred_ts(data_path: str, target: Optional[str] = None, step: int = 1, time_limit: float or int = 120):
    """

    :param data_path:
    :param target:
    :param step:
    :param time_limit:
    :return:
    """

    data = TimeSeriesDataFrame(data_path)
    print(data)
    # breakpoint()
    predictor = TimeSeriesPredictor(target=target, prediction_length=step,
                                    id_column="item_id",
                                    timestamp_column="timestamp").fit(data, time_limit=time_limit)
    predictions = predictor.predict(data)
    print(predictions)


def future_pred_tab(data_path_test: str, data_path_train: str, target='class', limit: float or int = 120):
    train_data = TabularDataset(data_path_train)
    test_data = TabularDataset(data_path_test)

    predictor = TabularPredictor(label=target).fit(train_data, time_limit=limit, )  # Fit models for 120s
    leaderboard = predictor.leaderboard(test_data)
    predictions = predictor.predict(test_data)
    print(f'{leaderboard}')
    print('________________________________________________________________\n\n')
    print(predictions)


if __name__ == '__main__':
    data_path = 'resample/resampled/01-16merged_logs_interpolated_resample_60_min_interval.csv'
    # future_pred_ts(data_path=data_path, target='DE_AREA')
    future_pred_tab(data_path, data_path, target='DE_AREA')
    # future_pred_ts(data_path=data_path)
