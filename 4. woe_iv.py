import pandas as pd
import numpy as np
import random

# 等频分箱
def bin_frequency(x, y, n=10): # x为待分箱的变量，y为target变量.n为分箱数量
    total = len(y) # 计算总样本数
    bad = y.sum()      # 计算坏样本数
    good = total - bad  # 计算好样本数
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.qcut(x,n)})  # 用pd.cut实现等频分箱
    d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin'])
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count() # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
    for i in range(len(d3)):

    d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
    for i in range(len(d3)):


    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
    print('分箱结果：')
    print(d4)
    print('IV值为：')
    print(iv)
    cut = [float('-inf')]
    for i in d4.min_bin:
        cut.append(i)
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe

# 等距分箱
def bin_distance(x, y, n=10): # x为待分箱的变量，y为target变量.n为分箱数量
    total = len(y)  # 计算总样本数
    bad = y.sum()      # 计算坏样本数
    good = total - bad  # 计算好样本数
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x,n)}) #利用pd.cut实现等距分箱
    d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin'])
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count() # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
    print('分箱结果：')
    print(d4)
    print('IV值为：')
    print(iv)
    cut = []
    cut.append(float('-inf'))
    for i in d4.min_bin:
        cut.append(i)
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe

# 卡方分箱
def chi_bins(df, col, target, confidence=3.841, bins=20):  # 设定自由度为1，卡方阈值为3.841，最大分箱数20
    total = df[target].count()  # 计算总样本数
    bad = df[target].sum()  # 计算坏样本总数
    good = total - bad  # 计算好样本总数
    total_bin = df.groupby([col])[target].count()  # 计算每个箱体总样本数
    total_bin_table = pd.DataFrame({'total': total_bin})  # 创建一个数据框保存结果
    bad_bin = df.groupby([col])[target].sum()  # 计算每个箱体的坏样本数
    bad_bin_table = pd.DataFrame({'bad': bad_bin})  # 创建一个数据框保存结果
    regroup = pd.merge(total_bin_table, bad_bin_table, left_index=True, right_index=True,
                       how='inner')  # 组合total_bin 和 bad_bin
    regroup.reset_index(inplace=True)
    regroup['good'] = regroup['total'] - regroup['bad']  # 计算每个箱体的好样本数
    regroup = regroup.drop(['total'], axis=1)  # 删除total
    np_regroup = np.array(regroup)  # 将regroup转为numpy

    # 处理连续没有正样本和负样本的区间，进行合并，以免卡方报错
    i = 0
    while i <= np_regroup.shape[0] - 2:
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (
                np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i -= 1
        i += 1

    # 对相邻两个区间的值进行卡方计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = ((np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 *
               (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2])) /\
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) *
               (np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)

    # 将卡方值最小的两个区间进行合并
    while 1:  # 除非设置break，否则会一直循环下去
        if len(chi_table) <= (bins - 1) or min(chi_table) >= confidence:
            break  # 当chi_table的值个数小于等于（箱体数-1) 或 最小的卡方值大于等于卡方阈值时，循环停止
        chi_min_index = np.argwhere(chi_table==min(chi_table))[0]  # 找出卡方最小值的索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, axis=0)

        if chi_min_index == np_regroup.shape[0] - 1:  # 当卡方最小值是最后两个区间时，计算当前区间和前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = ((np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                             np_regroup[chi_min_index - 1, 2]
                                             * np_regroup[chi_min_index, 1]) ** 2 * (
                                            np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]
                                            + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2])) / (
                                           (np_regroup[chi_min_index - 1, 1] +
                                            np_regroup[chi_min_index - 1, 2]) * (
                                           np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) *
                                           (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (
                                           np_regroup[chi_min_index - 1, 2] +
                                           np_regroup[chi_min_index, 2]))
            chi_table = np.delete(chi_table, chi_min_index, axis=0)  # 删除替换前的卡方值
        else:
            # 计算合并后当前区间和前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = ((np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                             np_regroup[chi_min_index - 1, 2]
                                             * np_regroup[chi_min_index, 1]) ** 2 * (
                                            np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]
                                            + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2])) / (
                                           (np_regroup[chi_min_index - 1, 1] +
                                            np_regroup[chi_min_index - 1, 2]) * (
                                           np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) *
                                           (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (
                                           np_regroup[chi_min_index - 1, 2] +
                                           np_regroup[chi_min_index, 2]))
            # 计算合并后当前区间和后一个区间的卡方值并替换
            chi_table[chi_min_index] = ((np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[
                chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 * (
                                        np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]
                                        + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2])) / (
                                       (np_regroup[chi_min_index, 1] +
                                        np_regroup[chi_min_index, 2]) * (
                                       np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) *
                                       (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (
                                       np_regroup[chi_min_index, 2] +
                                       np_regroup[chi_min_index + 1, 2]))
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)  # 删除替换前的卡方值


            # 将结果保存为一个数据框
    result_data = pd.DataFrame()
    result_data['col'] = [col] * np_regroup.shape[0]  # 结果第一列为变量名
    list_temp = []  # 创建一个空白的分组列
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:  # 当为第一个箱体时
            x = '0' + ',' + str(np_regroup[i, 0])
        elif i == np_regroup.shape[0] - 1:  # 当为最后一个箱体时
            x = str(np_regroup[i - 1, 0]) + '+'
        else:
            x = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
        list_temp.append(x)
    result_data['bin'] = list_temp
    result_data['bad'] = np_regroup[:, 1]
    result_data['good'] = np_regroup[:, 2]
    result_data['bad_rate'] = result_data['bad'] / total  # 计算每个箱体坏样本所占总样本比例
    result_data['badattr'] = result_data['bad'] / bad  # 计算每个箱体坏样本所占坏样本总数的比例
    result_data['goodattr'] = result_data['good'] / good  # 计算每个箱体好样本所占好样本总数的比例
    result_data['woe'] = np.log(result_data['goodattr'] / result_data['badattr'])  # 计算每个箱体的woe值
    iv = ((result_data['goodattr'] - result_data['badattr']) * result_data['woe']).sum()  # 计算每个变量的iv值
    print('分箱结果:')
    print(result_data)
    print('IV值为:')
    print(iv)
    return result_data, iv


# fianl one!!!!!!!!!
def merge_iv(df, y_label, box_name = 'frequency', n=10):
    labels = list(df.columns)
    labels.remove(y_label)

    if box_name == 'frequency':
        result_dict = {}
        for i in labels:
            result_dict[i] = bin_frequency(df[i], df[y_label], n)[1]
        return result_dict

    if box_name == 'distance':
        result_dict = {}
        for i in labels:
            result_dict[i] = bin_distance(df[i], df[y_label], n)[1]
        return result_dict



if __name__ == '__main__':
    print('a')
    # the bellows are for test
    x = np.array([x for x in range(1000)])
    x2 = np.array([x**2 for x in range(1000)])
    y = np.array([random.randint(0,1) for y in range(1000)])
    df0 = pd.DataFrame({'x1':x, 'x2':x2, 'y':y})
    d4, iv, cut, woe = bin_frequency(x, y)
    d4, iv, cut, woe = bin_distance(x, y, 20)
    data, iv = chi_bins(df0, 'x1', 'y')
    result1 = merge_iv(df0, 'y', box_name='frequency', n=20)
    result2 = merge_iv(df0, 'y', box_name='distance', n=20)
    set1 = set(x for x in range(10))
    # the above are for test
