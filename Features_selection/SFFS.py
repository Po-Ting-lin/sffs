import numpy as np
import tqdm
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
        print("Starting Sequential Floating Forward Selection... ")
        self.num_of_features = 0
        print("Number of features: {}".format(self.num_of_features))
        best_auc = 0
        status_outer = True
        while status_outer:

            # SFS algorithm
            self.tmp_features_set, this_auc = self.sfs(old_list=self.tmp_features_set)
            if this_auc > best_auc:
                best_auc = this_auc
            self.num_of_features += 1
            print("Number of features: {}".format(self.num_of_features))

            status_inner = True
            while status_inner:

                # check meet the requirement
                if self.num_of_features == self.k:
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
                        print("remove: {}".format(removed_feature))
                        self.set_verbose.append(after_leave)
                        self.auc_verbose.append(this_auc)
                        self.action_verbose.append("remove" + str(removed_feature).strip('{').strip('}'))
                        print("Number of features: {}".format(self.num_of_features))
                    else:
                        self.tmp_features_set = before_leave
                        status_inner = False

    def test_combination(self, test_feature_list):
        if self.predict_callback is None or len(test_feature_list) == 0:
            return 0.0
        else:
            return self.predict_callback(test_feature_list)

    def sfs(self, old_list):
        """ Sequential Feature Selection"""
        result = []
        record_list = []
        total_list = range(self.total_num_of_features)
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
        print("add: {}".format(added_features))
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
        plt.text(10, 0.92, s="best AUC: " + str(np.round(np.max(self.auc_verbose), 3)), bbox=props)
        plt.show()


