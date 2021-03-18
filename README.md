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

X: 88 cells * 12 features  
y: ground truth label

![](/demo_images/sffs.png)

```
the process of sffs:
10 -->
10, 11 -->
10, 11, 6 -->
10, 11, 6, 0 -->
10, 11, 6, 0, 2 -->
10, 11, 6, 0, 2, 8 -->
10, 11, 6, 2, 8 -->
10, 11, 6, 2, 8, 4 -->
...
...
```
The best performance is using the features \[10, 11, 6, 2, 8, 4].

## Reference
```
[1] Pudil, Pavel, Jana Novovičová, and Josef Kittler. 
    "Floating search methods in feature selection." 
    Pattern recognition letters 15.11 (1994): 1119-1125.
```

