"""
重庆位于东经105°17′至110°11′、北纬28°10′至32°13′之间
105.28333333333333
110.18333333333334
28.166666666666668
32.21666666666667
"""




import pandas as pd
import numpy as np
import copy
import random

random.seed(10)
# sample data
def get_sample_data():
    sample_data = np.random.rand(100, 2)
    return sample_data

# 获取经纬度的上下线
def high_low_bound(nparray, longitude_position, latitude_position):
    new_data = copy.deepcopy(nparray.T)
    longtitude_data = new_data[longitude_position]
    latitude_data = new_data[latitude_position]
    long_max = longtitude_data.max()
    long_min = longtitude_data.min()
    lati_max = latitude_data.max()
    lati_min = latitude_data.min()
    return long_max, long_min, lati_max, lati_min

# 获取某个点在哪个区域范围内
def get_between(value, low_bound, high_bound):
    if low_bound <= value <= high_bound:
        return True
    return False

# 第一种划分方式（可能未来有多种），给每个点打上标记（属于哪个格子）。
# 阅读m * n 矩阵方式阅读。
def blocks_cut_1(nparray, longitude_position, latitude_position, m_d, n_d):
    long_max, long_min, lati_max, lati_min = high_low_bound(nparray, longitude_position, latitude_position)
    long_len = (long_max - long_min) / n_d
    lati_len = (lati_max - lati_min) / m_d
    blocks_long_dict = {}
    blocks_lati_dict = {}
    for n in range(n_d):
        blocks_long_dict[str(n)] = [long_min + n * long_len, long_min + (n + 1) * long_len]
        # 计算过程需要修正，因为long_len, lati_len是保留的小数，会存在偏差
        if n == n_d - 1:
            blocks_long_dict[str(n)] = [long_min + n * long_len, long_max]

    for m in range(m_d):
        blocks_lati_dict[str(m)] = [lati_max - (m + 1) * lati_len, lati_max - m * lati_len]
        # 计算过程需要修正，因为long_len, lati_len是保留的小数，会存在偏差
        if m == m_d - 1:
            blocks_lati_dict[str(m)] = [lati_min, lati_max - m * lati_len]

    return blocks_long_dict, blocks_lati_dict

# 获取某个点的经度所属block
def longitude_block_id(value, long_block_dict):
    for block_id in long_block_dict:
        if get_between(value, long_block_dict[block_id][0], long_block_dict[block_id][1]):
            return block_id

# 获取某个点的维度所属block
def latitude_block_id(value, lati_block_dict):
    for block_id in lati_block_dict:
        if get_between(value, lati_block_dict[block_id][0], lati_block_dict[block_id][1]):
            return block_id


if __name__ == '__main__':
    print('a')
    test_data = get_sample_data()
    lon_max, lon_min, lat_max, lat_min = high_low_bound(test_data, 0, 1)
    blocks_long, blocks_lati = blocks_cut_1(test_data, 0, 1, 10, 10)
    long_lati_df = pd.DataFrame(test_data, columns=['long', 'lati'])
    long_lati_df['long_block'] = long_lati_df['long'].apply(lambda x: longitude_block_id(x, blocks_long))
    long_lati_df['lati_block'] = long_lati_df['lati'].apply(lambda x: latitude_block_id(x, blocks_lati))
