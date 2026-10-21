import logging
import os
import random

import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe

logger = logging.getLogger(__name__)


def load(config):
    data = {}
    problem = os.path.basename(config['data_dir'].rstrip('/\\'))
    train_file = os.path.join(config['data_dir'], f'{problem}_TRAIN.ts')
    test_file = os.path.join(config['data_dir'], f'{problem}_TEST.ts')
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f'Missing UEA files for {problem}. Expected:\n{train_file}\n{test_file}'
        )

    logger.info('Loading and preprocessing data ...')
    train_df, y_train_raw = load_from_tsfile_to_dataframe(train_file)
    test_df, y_test_raw = load_from_tsfile_to_dataframe(test_file)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    # Avoid test-set information leakage: sequence length is derived from TRAIN only.
    train_lengths = dataframe_lengths(train_df)
    max_seq_len = int(np.max(train_lengths))

    x_train_all = process_ts_data(train_df, max_seq_len, normalise=False)
    x_test = process_ts_data(test_df, max_seq_len, normalise=False)

    if config['val_ratio'] > 0:
        train_data, train_label, val_data, val_label = split_dataset(
            x_train_all, y_train, config['val_ratio'], seed=config.get('seed', 1234)
        )
    else:
        raise ValueError('val_ratio must be > 0 because the best checkpoint is selected on validation metrics')

    # Fit normalization on the training split only, then transform validation and test.
    if config['Norm']:
        mean, std = mean_std(train_data)
        mean = np.repeat(mean, max_seq_len).reshape(train_data.shape[1], max_seq_len)
        std = np.repeat(std, max_seq_len).reshape(train_data.shape[1], max_seq_len)
        train_data = mean_std_transform(train_data, mean, std)
        val_data = mean_std_transform(val_data, mean, std)
        x_test = mean_std_transform(x_test, mean, std)

    data['max_len'] = max_seq_len
    data['All_train_data'] = x_train_all
    data['All_train_label'] = y_train
    data['train_data'] = train_data
    data['train_label'] = train_label
    data['val_data'] = val_data
    data['val_label'] = val_label
    data['test_data'] = x_test
    data['test_label'] = y_test

    logger.info('%s samples will be used for training', len(train_label))
    logger.info('%s samples will be used for validation', len(val_label))
    logger.info('%s samples will be used for testing', len(y_test))
    return data


def dataframe_lengths(df):
    try:
        return df.map(lambda x: len(x)).values
    except AttributeError:
        return df.applymap(lambda x: len(x)).values


def split_dataset(data, label, validation_ratio, seed=1234):
    try:
        splitter = model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=validation_ratio, random_state=seed
        )
        train_indices, val_indices = next(splitter.split(X=np.zeros(len(label)), y=label))
    except ValueError:
        # Some tiny datasets/classes cannot satisfy stratified splitting. Use a deterministic random split instead.
        train_indices, val_indices = model_selection.train_test_split(
            np.arange(len(label)), test_size=validation_ratio, random_state=seed, shuffle=True
        )
    return data[train_indices], label[train_indices], data[val_indices], label[val_indices]


def fill_missing(x: np.ndarray, max_len: int, vary_len: str = 'suffix-noise', normalise: bool = True):
    if vary_len == 'zero':
        if normalise:
            x = StandardScaler().fit_transform(x)
        return np.nan_to_num(x)

    if vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = x[i, ~np.isnan(x[i])]
            if len(series) == 0:
                continue
            for j in range(max_len):
                x[i, j] = series[int(j * len(series) / max_len)]
            if normalise:
                x[i] = StandardScaler().fit_transform(x[i].reshape(-1, 1))[:, 0]
        return x

    for i in range(len(x)):
        valid = x[i, ~np.isnan(x[i])]
        if len(valid) == 0:
            x[i] = 0.0
        else:
            nan_mask = np.isnan(x[i])
            x[i, nan_mask] = [random.random() / 1000 for _ in range(np.sum(nan_mask))]
        if normalise:
            x[i] = StandardScaler().fit_transform(x[i].reshape(-1, 1))[:, 0]
    return x


def process_ts_data(x, max_len, vary_len: str = 'suffix-noise', normalise: bool = False):
    num_instances, num_dim = x.shape
    columns = x.columns
    output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)
    for i in range(num_dim):
        for j in range(num_instances):
            values = x[columns[i]][j].values
            end = min(len(values), max_len)
            output[j, i, :end] = values[:end]
            if end < max_len:
                output[j, i, end:] = np.nan
        output[:, i, :] = fill_missing(output[:, i, :], max_len, vary_len, normalise)
    return output


def mean_std(train_data):
    mean = np.mean(np.mean(train_data, axis=2), axis=0)
    std = np.max(np.std(train_data, axis=2), axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def mean_std_transform(data, mean, std):
    return (data - mean) / std
