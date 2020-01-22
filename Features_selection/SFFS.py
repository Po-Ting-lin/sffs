import numpy as np
import tqdm
from matplotlib import pyplot as plt

from configuration import *
from Machine_learning.ML_workflow import SupportVectorMachineWorkflow


class SequentialFloatingForwardSelection(object):
    def __init__(self, X, y, tag, require_d):
        # data
        self.X = X
        self.y = y
        self.tag = tag

        # parameters
        self.tem_features_set = []
        self.k = 0
        self.d = require_d

        # result
        self.set_verbose = []
        self.auc_verbose = []
        self.action_verbose = []
        self.check_input()

    def check_input(self):
        assert len(self.X.shape) == 2, "X input must be 2D!"
        assert len(self.y) == self.X.shape[0], "X len != y len!"

    def run(self):
        self.k = 0
        print("!!!!!!!!!!!!!!!!!current k: ", self.k, " !!!!!!!!!!!!!!!!!!!!!")
        best_auc = 0
        status_outer = True
        while status_outer:

            # SFS algorithm
            self.tem_features_set, this_auc = self.sfs(old_list=self.tem_features_set)
            if this_auc > best_auc:
                best_auc = this_auc
            self.k += 1
            print("!!!!!!!!!!!!!!!!!current k: ", self.k, " !!!!!!!!!!!!!!!!!!!!!")

            status_inner = True
            while status_inner:

                # check meet the requirement
                if self.k == self.d:
                    # yes
                    return best_auc
                else:
                    # no, so SBS algorithm
                    before_leave = self.tem_features_set
                    after_leave, this_auc = self.sbs(old_list=self.tem_features_set)

                    if this_auc > best_auc:
                        best_auc = this_auc
                        self.tem_features_set = after_leave
                        self.k -= 1
                        rm_num = set(after_leave).symmetric_difference(set(before_leave))
                        print("remove: ", rm_num)
                        self.set_verbose.append(after_leave)
                        self.auc_verbose.append(this_auc)
                        self.action_verbose.append("remove"+str(rm_num).strip('{').strip('}'))
                        print("!!!!!!!!!!!!!!!!!current k: ", self.k, " !!!!!!!!!!!!!!!!!!!!!")
                    else:
                        self.tem_features_set = before_leave
                        status_inner = False

    def sfs(self, old_list):
        """ Sequential Feature Selection"""
        result = []
        record_list = []
        total_list = range(len(delta_F))
        test_list = [ele for ele in total_list if ele not in old_list]
        # print("SFS: ", old_list)
        for t in tqdm.tqdm(test_list):
            if len(old_list) == 0:
                cur_list = [t]
            else:
                cur_list = old_list.copy()
                cur_list.append(t)
            cur_set = set(cur_list)
            record_list.append(cur_set)
            result.append(self.test_combination(cur_list))

        add_num = record_list[int(np.argmax(result))].symmetric_difference(set(old_list))
        print("add: ", add_num)
        self.set_verbose.append(list(record_list[int(np.argmax(result))]))
        self.auc_verbose.append(np.max(result))
        self.action_verbose.append("add"+str(add_num).strip('{').strip('}'))
        return list(record_list[int(np.argmax(result))]), np.max(result)

    def sbs(self, old_list):
        """Sequential Backward Selection"""
        assert len(old_list) > 0, "SBS: len(old list) > 0 !!!"
        result = []
        record_list = []
        # print("SBS: ", old_list)
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
        #################################################################################
        svmw = SupportVectorMachineWorkflow(self.X, self.y, self.tag, iters=500)
        svmw.run(choose_list=test_feature_list, print_mode=False)
        return svmw.auc

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


