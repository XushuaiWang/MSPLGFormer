import numpy as np
from sklearn import metrics


def print_confusion_matrix(conf_mat, label_strings=None, title='Confusion matrix'):
    if label_strings is None:
        label_strings = [str(i) for i in range(conf_mat.shape[0])]
    print(title)
    print('-' * len(title))
    print('true\\pred\t' + '\t'.join(label_strings))
    for label, row in zip(label_strings, conf_mat):
        print(str(label) + '\t' + '\t'.join(map(str, row)))


class Analyzer(object):
    def __init__(self, maxcharlength=35, plot=False, print_conf_mat=False, output_filepath=None):
        self.maxcharlength = maxcharlength
        self.plot = plot
        self.print_conf_mat = print_conf_mat

    def get_avg_prec_recall(self, conf_mat):
        pred_per_class = np.sum(conf_mat, axis=0)
        support = np.sum(conf_mat, axis=1)
        valid_precision = pred_per_class > 0
        if np.any(valid_precision):
            weights = support[valid_precision] / np.sum(support[valid_precision])
            precision_per_class = np.diag(conf_mat)[valid_precision] / pred_per_class[valid_precision]
            prec_avg = float(np.dot(weights, precision_per_class))
        else:
            prec_avg = 0.0
        rec_avg = float(np.trace(conf_mat) / np.sum(support)) if np.sum(support) > 0 else 0.0
        return prec_avg, rec_avg

    def analyze_classification(self, y_pred, y_true, class_names, excluded_classes=None):
        in_pred_labels = set(list(y_pred))
        in_true_labels = set(list(y_true))
        existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
        class_strings = [str(name) for name in class_names]
        existing_class_names = [class_strings[ind][:self.maxcharlength] for ind in existing_class_ind]

        conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=existing_class_ind)
        if self.print_conf_mat:
            print_confusion_matrix(conf_mat, label_strings=existing_class_names)

        precision, recall, f1, support = metrics.precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=existing_class_ind,
            zero_division=0,
        )
        total_accuracy = float(np.trace(conf_mat) / len(y_true)) if len(y_true) > 0 else 0.0
        prec_avg, rec_avg = self.get_avg_prec_recall(conf_mat)

        return {
            'total_accuracy': total_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'prec_avg': prec_avg,
            'rec_avg': rec_avg,
            'ConfMatrix': conf_mat,
        }
