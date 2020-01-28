# FeaturesSelection_sffs

According to the paper [Sequential Floating Forward Selection(SFFS)](https://www.sciencedirect.com/science/article/abs/pii/0167865594901279)
<br /> 
* Implement SFFS on SVC tasks.  

![](/demo_images/FSall.png)


## run test data
```python
from Features_selection import SequentialFloatingForwardSelection
from data_preparation import Preprocess
from configuration import DATE

# data preparation: using 88 cells
delta_X, y, tag = Preprocess(date_list=DATE, gamma_mode=True).delta_X_generator()

# SFFS
sffs = SequentialFloatingForwardSelection(delta_X, y, tag, require_d=12)
best_auc = sffs.run()
sffs.plot()
```

delta_X: 88 cells * 12 features after features scaling and gamma correction
y: ground truth label
tag: cells name

![](/demo_images/sffs.png)

```


```

