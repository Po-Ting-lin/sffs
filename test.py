import numpy as np
from svm_classification_for_testing.non_linear_svm import NonLinearSVM
from svm_classification_for_testing.non_linear_svm_mode import GammaMode, SvmClassificationMode
from Features_selection import SequentialFloatingForwardSelection

##############################################################################################
# replace with your predict function
# input: features index list
# output: metric of the prediction
def predict_callback(features_list):
    x = np.load("Features_selection/x.npy")
    y = np.load("Features_selection/y.npy")
    tag = np.load("Features_selection/tag.npy")
    cls = NonLinearSVM(iterations=300,
                       features_list=None,
                       model=None,
                       gamma_mode=GammaMode.CONSTANT_MODE,
                       classification_mode=SvmClassificationMode.TRAIN_WITH_DEFAULT,
                       pca_mode=False,
                       show_error_mode=False,
                       verbose=False)
    cls.set_features_list(features_list)
    cls.set_data_container(x, y, tag)
    return cls.main_classify_process()
##############################################################################################

total_number_of_features = 12
target_feature_number = 12
sffs = SequentialFloatingForwardSelection(total_number_of_features, target_feature_number, predict_callback)
sffs.process()
sffs.plot()













