## Introduction

According to the paper [Sequential Floating Forward Selection(SFFS)](https://www.sciencedirect.com/science/article/abs/pii/0167865594901279)

## Test result
```python
from Features_selection import SequentialFloatingForwardSelection

total_number_of_features = 12
target_feature_number = 12
sffs = SequentialFloatingForwardSelection(total_number_of_features, target_feature_number, predict_callback)
sffs.process()
sffs.plot()
```

```shell
Starting Sequential Floating Forward Selection...

Number of features: 1
        SFS:|############| --  sfs finish -- add: {10}

Number of features: 2
        SFS:|###########|  --  sfs finish -- add: {7}
        SBS:|##|           -- sbs finish
current best auc:  0.97 ; remove best auc:  0.923

Number of features: 3
        SFS:|##########|   --  sfs finish -- add: {3}
        SBS:|###|          -- sbs finish
current best auc:  0.973 ; remove best auc:  0.97

Number of features: 4
        SFS:|#########|    --  sfs finish -- add: {2}
        SBS:|####|         -- sbs finish
current best auc:  0.977 ; remove best auc:  0.973

Number of features: 5
        SFS:|########|     --  sfs finish -- add: {1}
        SBS:|#####|        -- sbs finish
current best auc:  0.979 ; remove best auc:  0.973

Number of features: 6
        SFS:|#######|      --  sfs finish -- add: {9}
        SBS:|#*****|       -- testing: { 2, 3, 7, 9, 10,  }
~

The best feature set: [1, 2, 3, 7, 10], auc: 0.979
```

## Plot
![](/demo_images/sffs.png)

## Reference
```
[1] Pudil, Pavel, Jana Novovičová, and Josef Kittler. 
    "Floating search methods in feature selection." 
    Pattern recognition letters 15.11 (1994): 1119-1125.
```

