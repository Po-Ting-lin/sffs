import sys
import numpy as np
from matplotlib import pyplot as plt


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
        self.set_verbose = []
        self.auc_verbose = []
        self.action_verbose = []

    def process(self):
        assert self.predict_callback is not None, "you need to input predict function"
        print("Starting Sequential Floating Forward Selection... ")
        self.num_of_features = 0
        best_auc = 0
        status_outer = True
        while status_outer:
            self.num_of_features += 1
            print("Number of features: {}".format(self.num_of_features))
            self.tmp_features_set, sfs_auc = self.sfs(old_list=self.tmp_features_set)
            best_auc = sfs_auc if sfs_auc > best_auc else best_auc

            if self.num_of_features == self.k:
                return best_auc
            status_inner = True
            while status_inner and len(self.tmp_features_set) > 1:
                sbs_auc, status_inner = self.sbs(old_list=self.tmp_features_set, prev_auc=best_auc)
                best_auc = sbs_auc if status_inner else best_auc
                print("current best auc: ", np.round(best_auc, 3), "; remove best auc: ", np.round(sbs_auc, 3))

    def test_combination(self, test_feature_list):
        if self.predict_callback is None or len(test_feature_list) == 0:
            return 0.0
        else:
            return self.predict_callback(test_feature_list)

    def sfs(self, old_list):
        """ Sequential Feature Selection"""
        result = []
        record_list = []
        test_list = [member for member in range(self.total_num_of_features) if member not in old_list]
        text = "\tSFS"
        for idx, t in enumerate(test_list):
            cur_list = np.append(np.array(old_list.copy(), dtype=int), t)
            record_list.append(set(cur_list))
            result.append(self.test_combination(cur_list))
            self.progress(text, idx, len(test_list), cur_list)

        added_feature, sfs_feature_list, sfs_best_auc = self.choose_best_auc(old_list, record_list, result)
        print("finish -- add: {}".format(added_feature))
        self.set_verbose.append(sfs_feature_list)
        self.auc_verbose.append(sfs_best_auc)
        self.action_verbose.append("add"+str(added_feature).strip('{').strip('}'))
        return sfs_feature_list, sfs_best_auc

    def sbs(self, old_list, prev_auc):
        """Sequential Backward Selection"""
        result = []
        record_list = []
        is_removed = False
        text = "\tSBS"
        for idx, t in enumerate(old_list):
            cur_list = old_list.copy()
            cur_list.remove(t)
            record_list.append(set(cur_list))
            result.append(self.test_combination(cur_list))
            self.progress(text, idx, len(old_list), cur_list)

        removed_feature, sbs_feature_list, sbs_best_auc = self.choose_best_auc(old_list, record_list, result)
        if sbs_best_auc > prev_auc:
            is_removed = True
            self.tmp_features_set = sbs_feature_list
            self.num_of_features -= 1
            self.set_verbose.append(sbs_feature_list)
            self.auc_verbose.append(sbs_best_auc)
            self.action_verbose.append("remove" + str(removed_feature).strip('{').strip('}'))
            print("finish -- remove: {}".format(removed_feature))
        else:
            print("finish")
        return sbs_best_auc, is_removed

    def choose_best_auc(self, old_features, new_features_list, auc_list):
        best_auc = np.max(auc_list)
        best_feature_set = new_features_list[int(np.argmax(auc_list))]
        diff_features = best_feature_set.symmetric_difference(set(old_features))
        return diff_features, list(best_feature_set), best_auc

    def progress(self, text, idx, test_size, cur_list):
        if test_size is 0:
            return
        sep = "" if len(cur_list) is 1 else ", "
        task = test_size - idx - 1
        text += ":|"
        for i in range(idx + 1):
            text += "#"
        for i in range(task):
            text += "*"
        text += "| -- "
        if idx + 1 is not test_size:
            text += "testing: { "
            for i in cur_list:
                text += str(i) + sep
            text += " }"
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

    def plot(self):
        string_set_verbose = [''.join(str(x).strip("[").strip("]")) for x in self.set_verbose]
        x = np.arange(len(self.auc_verbose))
        plt.figure(figsize=(10, 10))
        plt.plot(x, self.auc_verbose)
        plt.xticks(x, string_set_verbose, rotation=90)
        plt.title("Sequential Floating Forward Selection")
        plt.ylabel('AUC')
        plt.xlabel("Features")
        props = dict(boxstyle='round', facecolor='none', alpha=0.5)
        plt.text(10, 0.92, s="best AUC: " + str(np.round(np.max(self.auc_verbose), 3)), bbox=props)
        plt.show()


