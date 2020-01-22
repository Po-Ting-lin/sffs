from Features_selection import SequentialFloatingForwardSelection
from data_preparation import Preprocess
from configuration import DATE

delta_X, y, tag = Preprocess(date_list=DATE, gamma_mode=True).delta_X_generator()
sffs = SequentialFloatingForwardSelection(delta_X, y, tag, require_d=12)
best_auc = sffs.run()
# print(sffs.auc_verbose)
# print(sffs.action_verbose)
# print(best_auc)
sffs.plot()
######################################################################################################################






