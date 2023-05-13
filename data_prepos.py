from datetime import datetime

import pandas
from autogluon.timeseries import TimeSeriesDataFrame

op_dir = './resample'


def fix_format(csv_path='22-29-01-16merged_logs.csv'):
    """

    :param csv_path:
    :return:
    """

    raw_data = pandas.read_csv(f'{op_dir}/{csv_path}')
    time_label = 'timestamp'
    item_label = 'item_id'
    if time_label not in raw_data.columns and item_label not in raw_data.columns:
        raw_data.rename(columns={'timeStamp': 'timestamp'}, inplace=True)
        raw_data.insert(loc=0, column='item_id', value=0)

        stamps = raw_data['timestamp'].tolist()
        new_stamps = []
        for i in stamps:
            time_format = "%Y-%m-%d-%H-%M"
            normal_time = datetime.strptime(i, time_format)
            time_format = "%Y-%m-%d %H:%M:00"
            std_time_str = datetime.strftime(normal_time, time_format)
            new_stamps.append(std_time_str)
        raw_data['timestamp'] = new_stamps
    raw_data.to_csv(f'{op_dir}/{csv_path}', index=False)


def fix_value(csv_path: str = '22-29-01-16merged_logs.csv'):
    raw_data = TimeSeriesDataFrame(f'{op_dir}/{csv_path}')
    processed_data = raw_data.to_regular_index(freq='T')
    print(processed_data.freq, ' frequency')
    fixed_data = processed_data.fill_missing_values()
    temp = csv_path.replace('.csv', '')
    res_name = f'{temp}_interpolated'
    print(f'save to {res_name}')
    print(fixed_data)
    fixed_data.to_csv(f'{op_dir}/{res_name}.csv')
    return res_name


def resamp(original, sampling_steps=5, target_root='resample', original_root='./resample'):
    original_path = f'{original_root}/{original}.csv'
    print(f'loading at {original_path}')
    df = pandas.read_csv(original_path)
    print('wait to resamp', df)
    new_df = pandas.DataFrame(columns=df.columns)
    rows = len(df.index)
    for i in range(0, rows, sampling_steps):
        new_df.loc[i] = df.loc[i]
    save_path = f'{original_root}/{target_root}/{original}_resample_{sampling_steps}_min_interval.csv'
    print(new_df, f'\nsave at [{save_path}]')
    new_df.to_csv(save_path, index=False)


def prepos(original_file_path: str, sampling_steps=5):
    fix_format(original_file_path)
    interpolated = fix_value(original_file_path)
    print()
    resamp(interpolated, sampling_steps=sampling_steps)


if __name__ == '__main__':
    fname = ['16-18merged_log_test.csv',
             '16-19merged_logs.csv',
             '16-21merged_logs.csv']

    for f in fname:
        prepos(f, sampling_steps=20)
