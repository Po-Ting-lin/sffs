import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from svm_classification.non_linear_svm_mode import SvmClassificationMode
from svm_classification.classification import Classification, BinaryClassificationMetrics


class NonLinearSVM(Classification):
    def __init__(self, iterations, features_list, model, gamma_mode, classification_mode, pca_mode, show_error_mode, verbose=True):
        super().__init__()
        self.current_iter = None

        # customized input
        self.features_list = features_list
        self.model = model
        self.verbose = verbose

        # mode
        self.gamma_mode = gamma_mode
        self.classification_mode = SvmClassificationMode(classification_mode)
        self.pca_mode = pca_mode
        self.show_error_mode = show_error_mode

        # hyper parameter
        self.iterations = None
        self.gamma = None
        self.C = None
        self.positive_class_weight = None
        self.negative_class_weight = None
        self.gamma_range = np.logspace(-4, 1, 6)
        self.C_range = np.logspace(-2, 3, 6)
        self.positive_class_weight = 0.15
        self.negative_class_weight = 1 - self.positive_class_weight

        if self.classification_mode == SvmClassificationMode.PRE_TRAINED:
            self.model = model
            self.iterations = 1
        elif self.classification_mode == SvmClassificationMode.TRAIN_WITH_DEFAULT:
            self.iterations = iterations
            self.gamma = 0.1
            self.C = 10
        elif self.classification_mode == SvmClassificationMode.TRAIN_WITH_ADAPTIVE:
            self.iterations = iterations

        # data
        self.__tag = None
        self.__y = None
        self.__x = None
        self.__x_train = None
        self.__x_validation = None
        self.__x_test = None
        self.__y_train = None
        self.__y_validation = None
        self.__y_test = None
        self.__tag_test = None

    def set_data_container(self, x, y, tag):
        self.__x = x
        self.__y = y
        self.__tag = tag

    def reset_data_container(self):
        self.__x_train = None
        self.__x_validation = None
        self.__x_test = None
        self.__y_train = None
        self.__y_validation = None
        self.__y_test = None
        self.__tag_test = None

    def set_features_list(self, features_list):
        self.features_list = features_list

    def main_classify_process(self):
        self.__x = self.__choose_features()

        is_cell_in_test_set = []
        false_positive_verbose = []
        false_negative_verbose = []
        metrics = BinaryClassificationMetrics()
        metrics.reset_metrics()
        for i in range(self.iterations):
            self.reset_data_container()
            self.current_iter = i + 1
            if self.classification_mode == SvmClassificationMode.PRE_TRAINED:
                y_test_predicted, area_under_curve = self.__predict(self.__x, self.__y)
                self.__x_test = self.__x
                self.__y_test = self.__y
                self.__tag_test = self.__tag
            else:
                if self.classification_mode == SvmClassificationMode.TRAIN_WITH_ADAPTIVE:
                    self.__split_dataset()
                    self.__find_svm_parameter()
                elif self.classification_mode == SvmClassificationMode.TRAIN_WITH_DEFAULT:
                    self.__split_dataset()
                else:
                    raise Exception("Invalid svm_classification mode")
                self.model = self.__train(self.__x_train, self.__y_train, self.gamma, self.C, probability=True)
                y_test_predicted, area_under_curve = self.__predict(self.__x_test, self.__y_test)

            # show
            acc, recall, spec, package = show_result(self.__y_test, y_test_predicted)
            metrics.make_average_metrics([recall, spec, acc, area_under_curve] + list(package), i + 1)

            if self.classification_mode != SvmClassificationMode.PRE_TRAINED:
                if self.show_error_mode:
                    for ground_true, predicted_ans, tag in zip(self.__y_test, y_test_predicted, self.__tag_test):
                        is_cell_in_test_set.append(tag)
                        if ground_true != predicted_ans:
                            if ground_true == 0.0:
                                false_positive_verbose.append(tag)
                            elif ground_true == 1.0:
                                false_negative_verbose.append(tag)

        if self.show_error_mode and self.classification_mode != SvmClassificationMode.PRE_TRAINED:
            self.__plot_error_rate(is_cell_in_test_set, false_negative_verbose, false_positive_verbose)
        return metrics.area_under_curve

    def __choose_features(self):
        if self.features_list is None or len(self.features_list) == 0:
            return self.__x
        else:
            delta_x_chose = []
            for ch in self.features_list:
                if len(delta_x_chose) == 0:
                    delta_x_chose = np.array(self.__x[:, ch])
                else:
                    delta_x_chose = np.vstack((delta_x_chose, self.__x[:, ch]))
            delta_x_chose = delta_x_chose.T
            if len(delta_x_chose.shape) == 1:
                delta_x_chose = delta_x_chose.reshape((-1, 1))
            return delta_x_chose

    def __find_svm_parameter(self):
        pass

    def __train(self, x_train, y_train, gamma, c, probability=True):
        clf = SVC(kernel='rbf', gamma=gamma, C=c,
                  class_weight={0: self.negative_class_weight, 1: self.positive_class_weight},
                  probability=probability)
        clf.fit(x_train, y_train)
        return clf

    def __predict(self, x_test, y_test):
        yp_test_score = self.model.predict_proba(x_test)[:, 1]
        yp_test_pred = yp_test_score.copy()
        pred_threshold, area_under_curve = find_balance_recall_specificity(y_test, yp_test_score, plot_mode=False)
        yp_test_pred[yp_test_score > pred_threshold] = 1.0
        yp_test_pred[yp_test_score <= pred_threshold] = 0.0
        return yp_test_pred, area_under_curve

    def __split_dataset(self):
        def number_of_positive(y_arr):
            return int(len(y_arr) - np.sum(y_arr)), int(np.sum(y_arr))

        # train val test = 0.4: 0.2: 0.4
        self.__x_train, self.__x_test, tag_train, self.__tag_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__tag, self.__y, test_size=0.4, stratify=self.__y)  # 0.4
        if self.classification_mode == SvmClassificationMode.TRAIN_WITH_ADAPTIVE:
            self.__x_train, self.__x_validation, _, _, self.__y_train, self.__y_validation = train_test_split(self.__x_train, tag_train, self.__y_train, test_size=0.2, stratify=self.__y_train)  # 0.3
            self.__assert_y(self.__y_validation)
        elif self.classification_mode == SvmClassificationMode.TRAIN_WITH_DEFAULT:
            pass
        self.__assert_y(self.__y_train)
        self.__assert_y(self.__y_test)

    def __assert_y(self, arr):
        if np.sum(arr) == 0.0:
            raise Exception("All the y is negative")
        if np.sum(arr) == len(arr):
            raise Exception("All the y is positive")

    def __plot_error_rate(self, is_cell_in_test_set, false_negative_verbose, false_positive_verbose):
        test_count_dict = Counter(is_cell_in_test_set)
        fn_dict = Counter(false_negative_verbose)
        fp_dict = Counter(false_positive_verbose)
        print(test_count_dict)
        print("FN: ", fn_dict)
        print("FP: ", fp_dict)

        fn_dict_ratio, fp_dict_ratio = dict(), dict()
        for k, v in fn_dict.items():
            fn_dict_ratio.update({k: v / test_count_dict[k]})
        for k, v in fp_dict.items():
            fp_dict_ratio.update({k: v / test_count_dict[k]})
        print("FN: ", fn_dict_ratio)
        print("FP: ", fp_dict_ratio)

        fn_dict_ratio = OrderedDict(sorted(fn_dict_ratio.items(), key=lambda t: t[1]))
        fn_dict_ratio = OrderedDict(reversed(list(fn_dict_ratio.items())))
        fp_dict_ratio = OrderedDict(sorted(fp_dict_ratio.items(), key=lambda t: t[1]))
        fp_dict_ratio = OrderedDict(reversed(list(fp_dict_ratio.items())))
        plt.figure()
        plt.title("False negative cells")
        plt.bar(range(len(fn_dict_ratio)), list(fn_dict_ratio.values()), align='center')
        plt.xticks(range(len(fn_dict_ratio)), list(fn_dict_ratio.keys()), rotation=90)
        plt.ylabel("probability")
        plt.show()
        plt.figure()
        plt.title("False positive cells")
        plt.bar(range(len(fp_dict_ratio)), list(fp_dict_ratio.values()), align='center')
        plt.xticks(range(len(fp_dict_ratio)), list(fp_dict_ratio.keys()), rotation=90)
        plt.ylabel("probability")
        plt.show()


def show_result(yp_test, yp_test_pred):
    tn, fp, fn, tp = confusion_matrix(yp_test, yp_test_pred).ravel()
    Recall = tp / (tp + fn)
    Specificity = tn / (tn + fp)
    A = (tp + tn) / (tp + fp + fn + tn)
    cm_paclage = tn, fp, fn, tp
    return A, Recall, Specificity, cm_paclage


def find_balance_recall_specificity(yp_test, yp_test_score, plot_mode=False, print_mode=False):
    FPR, TPR, threshold = roc_curve(yp_test, yp_test_score)
    area_under_curve = auc(FPR, TPR)

    yp_test_pred = np.zeros(len(yp_test_score), dtype=float)
    gap = 1
    best_t = 0.5
    for t in threshold:
        yp_test_pred[yp_test_score > t] = 1.0
        yp_test_pred[yp_test_score <= t] = 0.0
        tn, fp, fn, tp = confusion_matrix(yp_test, yp_test_pred).ravel()
        Recall = tp / (tp + fn)
        Specificity = tn / (tn + fp)
        if np.abs(Recall - Specificity) < gap:
            gap = np.abs(Recall - Specificity)
            best_t = t
    if print_mode:
        print("best threshold:", best_t)

    if plot_mode:
        plt.figure(figsize=(6, 6), dpi=100)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(FPR, TPR, label='Area under curve = {:.3f}'.format(area_under_curve))
        for i in range(len(FPR)):
            plt.scatter(FPR[i], TPR[i], s=8, c='r')
            FPR[i] -= 0.001
            TPR[i] -= 0.04
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.xlim(-0.005, 1.005)
        plt.ylim(-0.005, 1.005)
        plt.title('ROC curve')
        plt.legend(loc='lower right')
        plt.show()
    if print_mode:
        print("AUC: ", round(area_under_curve, 3))
    return best_t, area_under_curve