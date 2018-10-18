import pandas as pd

"""
sample:
'2017-07-01 00:00:00'
type: str


目标：
0. 针对目标：生成时间
1. 按照时间排序
2. 针对排序后的时间计算时间差。
3. 索取当天内的某个时间段数据。
"""

from datetime import datetime
from datetime import timedelta


time1 = '2017-07-01 00:00:00'

time2 = '2017-07-12 12:00:00'

tt1 = datetime.strptime(time1, '%Y-%m-%d %H:%M:%S')

tt2 = tt1 + timedelta(12)

tt3 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')

delta = tt3 - tt1

delta.days
