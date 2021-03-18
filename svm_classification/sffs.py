import numpy as np
import tqdm
from matplotlib import pyplot as plt
from config.configuration import *
from core.svm_classification.non_linear_svm import NonLinearSVM
from core.svm_classification.non_linear_svm_mode import GammaMode, SvmClassificationMode


class SequentialFloatingForwardSelection(NonLinearSVM):
    def __init__(self, date_list, target_number_of_features, verbose=True):
        super().__init__(iterations=300,
                         date_list=date_list,
                         features_list=None,
                         model=None,
                         gamma_mode=GammaMode.CONSTANT_MODE,
                         classification_mode=SvmClassificationMode.TRAIN_WITH_DEFAULT,
                         pca_mode=False,
                         show_error_mode=False,
                         verbose=False)

        # mode
        self.verbose = verbose

        # parameters
        self.tmp_features_set = []
        self.num_of_features = 0
        self.target_number_of_features = target_number_of_features

        # result
        self.set_verbose = []
        self.auc_verbose = []
        self.action_verbose = []

    def process(self):
        self.logs.debug("Starting Sequential Floating Forward Selection... ")
        if not self.verbose:
            self.logs.disabled = True
        self.num_of_features = 0
        self.logs.debug("Number of features: {}".format(self.num_of_features))
        best_auc = 0
        status_outer = True
        while status_outer:

            # SFS algorithm
            self.tmp_features_set, this_auc = self.sfs(old_list=self.tmp_features_set)
            if this_auc > best_auc:
                best_auc = this_auc
            self.num_of_features += 1
            self.logs.debug("Number of features: {}".format(self.num_of_features))

            status_inner = True
            while status_inner:

                # check meet the requirement
                if self.num_of_features == self.target_number_of_features:
                    # yes
                    return best_auc
                else:
                    # no, so SBS algorithm
                    before_leave = self.tmp_features_set
                    after_leave, this_auc = self.sbs(old_list=self.tmp_features_set)

                    if this_auc > best_auc:
                        best_auc = this_auc
                        self.tmp_features_set = after_leave
                        self.num_of_features -= 1
                        removed_feature = set(after_leave).symmetric_difference(set(before_leave))
                        self.logs.debug("remove: {}".format(removed_feature))
                        self.set_verbose.append(after_leave)
                        self.auc_verbose.append(this_auc)
                        self.action_verbose.append("remove" + str(removed_feature).strip('{').strip('}'))
                        self.logs.debug("Number of features: {}".format(self.num_of_features))
                    else:
                        self.tmp_features_set = before_leave
                        status_inner = False
        if not self.verbose:
            self.logs.disabled = False

    def sfs(self, old_list):
        """ Sequential Feature Selection"""
        result = []
        record_list = []
        total_list = range(len(DELTA_FEATURES))
        test_list = [member for member in total_list if member not in old_list]
        for t in tqdm.tqdm(test_list):
            if len(old_list) == 0:
                cur_list = [t]
            else:
                cur_list = old_list.copy()
                cur_list.append(t)
            cur_set = set(cur_list)
            record_list.append(cur_set)
            result.append(self.test_combination(cur_list))

        added_features = record_list[int(np.argmax(result))].symmetric_difference(set(old_list))
        self.logs.debug("add: {}".format(added_features))
        self.set_verbose.append(list(record_list[int(np.argmax(result))]))
        self.auc_verbose.append(np.max(result))
        self.action_verbose.append("add"+str(added_features).strip('{').strip('}'))
        return list(record_list[int(np.argmax(result))]), np.max(result)

    def sbs(self, old_list):
        """Sequential Backward Selection"""
        assert len(old_list) > 0, "SBS: len(old list) > 0 !!!"
        result = []
        record_list = []
        for t in tqdm.tqdm(old_list):
            cur_list = old_list.copy()
            cur_list.remove(t)
            cur_set = set(cur_list)
            record_list.append(cur_set)
            result.append(self.test_combination(cur_list))
        return list(record_list[int(np.argmax(result))]), np.max(result)

    def test_combination(self, test_feature_list):
        if len(test_feature_list) == 0:
            return 0.0
        self.set_features_list(test_feature_list)
        self.set_data_container()
        return self.main_classify_process()

    def plot(self):
        string_set_verbose = [''.join(str(x).strip("[").strip("]")) for x in self.set_verbose]
        x = np.arange(len(self.auc_verbose))
        plt.figure(figsize=(8, 8))
        plt.plot(x, self.auc_verbose)
        plt.xticks(x, string_set_verbose, rotation=90)
        plt.title("Sequential Floating Forward Selection")
        plt.ylabel('AUC')
        plt.xlabel("Features")
        props = dict(boxstyle='round', facecolor='none', alpha=0.5)
        plt.text(10, 0.92, s="best AUC: " + str(np.max(self.auc_verbose)), bbox=props)
        plt.show()

