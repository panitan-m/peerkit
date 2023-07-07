import logging
import math
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error, accuracy_score

logger = logging.getLogger(__name__)

def rmse(y, y_):
    return math.sqrt(mean_squared_error(y, y_))


def evaluate_mean(train_dataset, test_dataset, aspects):
    evaluate_mean = []
    evaluate_major = []
    for y_aspect_train, y_aspect_test in zip(train_dataset.y.T, test_dataset.y.T):
        mean_aspect = np.average(y_aspect_train)
        major_aspect = Counter(y_aspect_train).most_common(1)[0][0]
        evaluate_mean_aspect = rmse(y_aspect_test, [mean_aspect]*len(y_aspect_test))
        evaluate_major_aspect = rmse(y_aspect_test, [major_aspect]*len(y_aspect_test))
        evaluate_mean.append(evaluate_mean_aspect)
        evaluate_major.append(evaluate_major_aspect)
    logger.info('Majority (Test)')
    for mean, major,a in zip(evaluate_mean, evaluate_major, aspects):
        logger.info('\t%25s\t%.4f\t%.4f'%(a.upper(), mean, major))
    logger.info('\t%25s\t%.4f\t%.4f'%('TOTAL',np.average(evaluate_mean), np.average(evaluate_major)))
    
    
def evaluate_major(train_dataset, test_dataset, aspects):
    evaluate_acc = []
    for y_aspect_train, y_aspect_test in zip(train_dataset.y.T, test_dataset.y.T):
        major_aspect = Counter(y_aspect_train).most_common(1)[0][0]
        evaluate_aspect_acc = accuracy_score(y_aspect_test, [major_aspect]*len(y_aspect_test))
        evaluate_acc.append(evaluate_aspect_acc)
    logger.info('Majority (Test)')
    for acc, a in zip(evaluate_acc, aspects):
        logger.info('\t%25s\t%.4f'%(a.upper(), acc))
    logger.info('\t%25s\t%.4f'%('TOTAL',np.average(evaluate_acc)))