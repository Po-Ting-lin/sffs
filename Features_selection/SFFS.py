import sys
import numpy as np
from matplotlib import pyplot as plt


class FeatureSet(object):
    def __init__(self, f_set, auc):
        self.f_set = f_set
        self.auc = auc

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return "feature set: {}, auc: {}".format(self.f_set, self.auc)


class SequentialFloatingForwardSelection(object):
    def __init__(self, total_num_of_features, k, predict_callback=None):
        # parameters
        self.tmp_features_set = []
        self.total_num_of_features = total_num_of_features
        self.num_of_features = 0
        self.k = k

        # predict callback method
        self.predict_callback = predict_callback

        # result
        self.best_feature_set = None
        self.f_set_list = []
        self.action_verbose = []

    def process(self):
        assert self.predict_callback is not None, "you need to input predict function"
        print("Starting Sequential Floating Forward Selection... ")
        self.num_of_features = 0
        self.best_feature_set = FeatureSet([], 0)
        status_outer = True
        while status_outer:
            self.num_of_features += 1
            print("\nNumber of features: {}".format(self.num_of_features))
            sfs_set = self.__sfs(old_list=self.tmp_features_set)
            self.best_feature_set = sfs_set if sfs_set > self.best_feature_set else self.best_feature_set

            if self.num_of_features == self.k:
                print("The best", self.best_feature_set)
                return self.best_feature_set
            status_inner = True
            while status_inner and len(self.tmp_features_set) > 1:
                sbs_set, status_inner = self.__sbs(old_list=self.tmp_features_set, prev_auc=self.best_feature_set.auc)
                self.best_feature_set = sbs_set if status_inner else self.best_feature_set
                print("current best auc: ", np.round(self.best_feature_set.auc, 3), "; remove best auc: ", np.round(sbs_set.auc, 3))

    def plot(self):
        f_set_list = [e.f_set for e in self.f_set_list]
        auc_list = [e.auc for e in self.f_set_list]
        string_set_verbose = [''.join(str(x).strip("[").strip("]")) for x in f_set_list]
        x = np.arange(len(auc_list))
        plt.figure(figsize=(10, 10))
        plt.plot(x, auc_list)
        plt.xticks(x, string_set_verbose, rotation=90)
        plt.title("Sequential Floating Forward Selection")
        plt.ylabel('AUC')
        plt.xlabel("Features")
        props = dict(boxstyle='round', facecolor='none', alpha=0.5)
        plt.text(10, 0.92, s="best AUC: " + str(np.round(self.best_feature_set.auc, 3)), bbox=props)
        plt.show()

    def __test_combination(self, test_feature_list):
        if self.predict_callback is None or len(test_feature_list) == 0:
            return 0.0
        else:
            return self.predict_callback(test_feature_list)

    def __sfs(self, old_list):
        """ Sequential Feature Selection"""
        result = []
        record_list = []
        test_list = [member for member in range(self.total_num_of_features) if member not in old_list]
        text = "\tSFS"
        for idx, t in enumerate(test_list):
            cur_list = np.append(np.array(old_list.copy(), dtype=int), t)
            record_list.append(set(cur_list))
            result.append(self.__test_combination(cur_list))
            self.__progress_bar(text, idx, len(test_list), cur_list)

        added_feature, sfs_feature_list, sfs_best_auc = self.__choose_best_auc(old_list, record_list, result)
        print(" sfs finish -- add: {}".format(added_feature) + " " * 40)
        sfs_set = FeatureSet(sfs_feature_list, sfs_best_auc)
        self.tmp_features_set = sfs_feature_list
        self.f_set_list.append(sfs_set)
        self.action_verbose.append("add"+str(added_feature).strip('{').strip('}'))
        return sfs_set

    def __sbs(self, old_list, prev_auc):
        """Sequential Backward Selection"""
        result = []
        record_list = []
        is_removed = False
        text = "\tSBS"
        for idx, t in enumerate(old_list):
            cur_list = old_list.copy()
            cur_list.remove(t)
            record_list.append(set(cur_list))
            result.append(self.__test_combination(cur_list))
            self.__progress_bar(text, idx, len(old_list), cur_list)

        removed_feature, sbs_feature_list, sbs_best_auc = self.__choose_best_auc(old_list, record_list, result)
        sbs_set = FeatureSet(sbs_feature_list, sbs_best_auc)
        if sbs_best_auc > prev_auc:
            is_removed = True
            self.tmp_features_set = sbs_feature_list
            self.num_of_features -= 1
            self.f_set_list.append(FeatureSet(sbs_feature_list, sbs_best_auc))
            self.action_verbose.append("remove" + str(removed_feature).strip('{').strip('}'))
            print(" sbs finish -- remove: {}".format(removed_feature) + " " * 40)
        else:
            print("sbs finish" + " " * 40)
        return sbs_set, is_removed

    def __choose_best_auc(self, old_features, new_features_list, auc_list):
        best_auc = np.max(auc_list)
        best_feature_set = new_features_list[int(np.argmax(auc_list))]
        diff_features = best_feature_set.symmetric_difference(set(old_features))
        return diff_features, list(best_feature_set), best_auc

    def __progress_bar(self, text, idx, test_size, cur_list):
        if test_size is 0:
            return
        sep = "" if len(cur_list) is 1 else ", "
        task = test_size - idx - 1
        space = self.total_num_of_features - test_size
        text += ":|"
        for i in range(idx + 1):
            text += "#"
        for i in range(task):
            text += "*"
        text += "|"
        for i in range(space):
            text += " "
        text += " -- "
        if idx + 1 is not test_size:
            text += "testing: { "
            for i in cur_list:
                text += str(i) + sep
            text += " }"
        sys.stdout.write("\r" + text)
        sys.stdout.flush()


