import numpy as np


def naive_MASE(pred, test_true, train_true):
    pred = np.array(pred)
    test_true = np.array(test_true)
    train_true = np.array(train_true)

    MAE = np.mean(np.abs(pred - test_true))
    naive_MAE = np.mean(np.abs(train_true[1:] - train_true[:-1]))
    return MAE / naive_MAE


def seasonal_MASE(pred, test_true, train_true, seasonality=12):
    pred = np.array(pred)
    test_true = np.array(test_true)
    train_true = np.array(train_true)
    MAE = np.mean(np.abs(pred - test_true))
    naive_MAE = np.mean(np.abs(train_true[seasonality:] - train_true[:-seasonality]))
    return MAE / naive_MAE


def SMAPE(pred, true):
    numerator = np.abs(pred - true)
    denominator = np.abs(true) + np.abs(pred)
    smape = 200.0 * np.mean(numerator / denominator)
    return smape
