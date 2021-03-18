import numpy as np
from svm_classification.non_linear_svm import NonLinearSVM
from svm_classification.non_linear_svm_mode import GammaMode, SvmClassificationMode
from Features_selection import SequentialFloatingForwardSelection


# customized predict function
def predict_callback(features_list):
    x = np.load("x.npy")
    y = np.load("y.npy")
    tag = np.load("tag.npy")
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


total_number_of_features = 12
target_number = 12
sffs = SequentialFloatingForwardSelection(total_number_of_features, target_number, predict_callback)
sffs.process()
sffs.plot()

######################################################################################################################






