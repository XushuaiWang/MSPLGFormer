import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__author__ = "Chang Wei Tan"


# Much of this code is adapted from https://github.com/hfawaz/dl-4-tsc

def prepare_inputs_deep_learning(train_inputs, test_inputs, window_len=40, stride=20,
                                 val_size=1, random_state=1234, verbose=1):
    """Split training series and extract fixed-length windows from data/label frames.

    val_size is the number of training series held out using random_state.
    Returns train, validation, and test arrays shaped [windows, window_len, channels]
    and their majority-class labels. Validation arrays are None when there are
    too few training series to hold out val_size series. verbose controls progress output.
    """
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    if len(train_inputs) > val_size:
        train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                    test_size=val_size,
                                                    random_state=random_state)
    else:
        train_series = range(len(train_inputs))
        val_series = None

    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if val_series is None:
        X_val = None
        y_val = None
    else:
        X_val = []
        y_val = []
        for i in val_series:
            this_series = train_inputs.data[i]
            this_series_labels = train_inputs.label[i]
            subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                           window_size=window_len,
                                                           stride=stride)
            [X_val.append(x) for x in subsequences]
            [y_val.append(x) for x in sub_label]
        X_val = np.array(X_val)
        y_val = np.array(y_val)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_subsequences(X_data, y_data, window_size=30, stride=1, norm=False):
    """Extract complete windows and assign each window its majority-class label.

    X_data has shape [time, channels]; y_data contains one class label per timepoint.
    """
    data_len, data_dim = X_data.shape
    subsequences = []
    labels = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end > data_len:
            break
        tmp = X_data[i:end, :]
        if norm:
            # Standardize each channel within this window when requested.
            scaler = StandardScaler()
            tmp = scaler.fit_transform(tmp)
        subsequences.append(tmp)
        label = np.asarray(stats.mode(y_data[i:end], axis=None).mode).item()
        labels.append(label)
        count += 1
    return np.array(subsequences), np.array(labels)
