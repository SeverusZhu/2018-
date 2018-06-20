# coding: utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame

lgb_sub = pd.read_csv("../buptloc/sub_lgb_12_0527_1140.csv")

lgb_sub = DataFrame(lgb_sub)

for j in lgb_sub.columns:
    for k in range(len(lgb_sub)):
        if lgb_sub[j][k] < 0:
            lgb_sub[j][k] = - lgb_sub[j][k]

lgb_sub.to_csv("../buptloc/sub_lgb_12_0527_1140.csv")
