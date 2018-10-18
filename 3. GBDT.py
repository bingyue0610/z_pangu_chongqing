"""
设计流程思考：
0. 先完成模型预设
1. 先完成二分类的各种最优找到各种最优，存在dict里面
2. 再设计调优步骤。
3. 最后设计cv步骤。
"""
from sklearn import datasets
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier



def personalize_evaluation(y_true, y_pred, top_num):
    """

    :param y_true: 必须是0,1分类 np.array
    :param y_pred:
    :param top_num: 分数最高的前n位置
    :return: top_num_positive_num_percent: 前100的true_positive在前100的比例。
             top_num_positive_overall_positive_percent: 前100的true_positive在所有test集中的比例
    """
    index_value_dict = {index: y_pred[index] for index in range(len(y_pred))}
    index_value_list_top_num = sorted(index_value_dict.items(), key=lambda d:d[1], reverse=True)[:top_num]
    top_num_true_poistive = sum([ y_true[element[0]] for element in index_value_list_top_num])
    print(top_num_true_poistive)
    top_num_positive_num_percent = top_num_true_poistive / top_num
    top_num_positive_overall_positive_percent = top_num_true_poistive / sum(y_pred)
    return top_num_positive_num_percent, top_num_positive_overall_positive_percent


def cross_validation(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type):
    """
    :param gbdt_model:gbdt_model
    :param n_split: k for k_fold
    :param n_repeat: 分几轮
    :param a_x: 所有的x
    :param a_y: 所有的y
    :param top_num: 前多少个num
    :param estimation_type: "top_num_positive_over_num", "top_num_positive_over_all"
    :return: estimation list 的平均分
    """
    estimation_list = []
    tmp_rkf = RepeatedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=2652124)
    for tmp_train_index, tmp_test_index in tmp_rkf.split(a_x):
        x_train, x_test = a_x[tmp_train_index], a_x[tmp_test_index]
        y_train, y_test = a_y[tmp_train_index], a_y[tmp_test_index]
        gbdt_model.fit(x_train, y_train)
        y_pred = grd.predict(x_test)
        if estimation_type == 'top_num_positive_over_num':
            estimation_list.append(personalize_evaluation(y_test, y_pred, top_num)[0])
        elif estimation_type == 'top_num_positive_over_all':
            estimation_list.append(personalize_evaluation(y_test, y_pred, top_num)[0])
        print(estimation_list)
    return sum(estimation_list) / len(estimation_list)


def choose_best_loss(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type):
    """
    :param gbdt_model:gbdt_model
    :param n_split: k for k_fold
    :param n_repeat: 分几轮
    :param a_x: 所有的x
    :param a_y: 所有的y
    :param top_num: 前多少个num
    :param estimation_type: "top_num_positive_over_num", "top_num_positive_over_all"
    :return: 最好参数的model
    """
    result_dict = {}
    try:
        gbdt_model.set_params(loss='exponential')
        result_dict['exponential'] = cross_validation(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type)
    except:
        print('wrong on: loss = exponential')

    try:
        gbdt_model.set_params(loss='deviance')
        result_dict['deviance'] = cross_validation(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type)
    except:
        print('wrong on: loss = deviance')

    tmp_score = -1
    result_list = []
    for key in result_dict:
        if result_dict[key] > tmp_score:
            result_list = [key, result_dict[key]]
            tmp_score = result_dict[key]
    gbdt_model.set_params(loss=result_list[0])
    print('best score is:', result_list[1])
    print("the best loss is:", result_list[0])
    return gbdt_model


def choose_best_learning_rate(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type, start_num, end_num, steps):
    """
    :param gbdt_model:gbdt_model
    :param n_split: k for k_fold
    :param n_repeat: 分几轮
    :param a_x: 所有的x
    :param a_y: 所有的y
    :param top_num: 前多少个num
    :param estimation_type: "top_num_positive_over_num", "top_num_positive_over_all"
    :param start_num: 起点（取不到起点，只能渠道起点+1个steps）
    :param end_num: 终点
    :param steps: 增长步长
    :return: 最好参数的model
    """
    result_dict = {}
    set_rate = start_num
    while set_rate < end_num:
        set_rate += steps
        try:
            gbdt_model.set_params(learning_rate=set_rate)
            result_dict[set_rate] = cross_validation(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type)
        except:
            print('wrong on: learning_rate = %s' %set_rate)

    tmp_score = -1
    result_list = []
    for key in result_dict:
        if result_dict[key] > tmp_score:
            result_list = [key, result_dict[key]]
            tmp_score = result_dict[key]
    gbdt_model.set_params(learning_rate=result_list[0])
    print('best score is:', result_list[1])
    print('the best learning_rate is:', result_list[0])
    return gbdt_model


def choose_best_n_estimators(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type, start_num, end_num, steps):
    """
    :param gbdt_model:gbdt_model
    :param n_split: k for k_fold
    :param n_repeat: 分几轮
    :param a_x: 所有的x
    :param a_y: 所有的y
    :param top_num: 前多少个num
    :param estimation_type: "top_num_positive_over_num", "top_num_positive_over_all"
    :param start_num: 起点（取不到起点，只能渠道起点+1个steps）
    :param end_num: 终点
    :param steps: 增长步长
    :return: 最好参数的model
    """
    result_dict = {}
    set_estimators = start_num
    while set_estimators < end_num:
        set_estimators += steps
        try:
            gbdt_model.set_params(n_estimators=set_estimators)
            result_dict[set_estimators] = cross_validation(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type)
        except:
            print('wrong on: n_estimators = %s' % set_estimators)

    tmp_score = -1
    result_list = []
    for key in result_dict:
        if result_dict[key] > tmp_score:
            result_list = [key, result_dict[key]]
            tmp_score = result_dict[key]
    gbdt_model.set_params(n_estimators=result_list[0])
    print('best score is:', result_list[1])
    print('the best n_estimators is:', result_list[0])
    return gbdt_model


def choose_best_max_depth(gbdt_model, n_split, n_repeat, a_x, a_y, top_num, estimation_type, start_num, end_num, steps):
    """
    :param gbdt_model:gbdt_model
    :param n_split: k for k_fold
    :param n_repeat: 分几轮
    :param a_x: 所有的
    :param a_y: 所有的y
    :param top_num: 前多少个num
    :param estimation_type: "top_num_positive_over_num", "top_num_positive_over_all"
    :param start_num: 起点（取不到起点，只能渠道起点+1个steps）
    :param end_num: 终点
    :param steps: 增长步长
    :return: 最好参数的model
    """
    result_dict = {}
    set_max_depth = start_num
    while set_max_depth < end_num:
        set_max_depth += steps
        try:
            gbdt_model.set_params(max_depth=set_max_depth)
            result_dict[set_max_depth] = cross_validation(gbdt_model, n_split, n_repeat, a_x, a_y, top_num,
                                                           estimation_type)
        except:
            print('wrong on: max_depth = %s' % set_max_depth)

    tmp_score = -1
    result_list = []
    for key in result_dict:
        if result_dict[key] > tmp_score:
            result_list = [key, result_dict[key]]
            tmp_score = result_dict[key]
    gbdt_model.set_params(max_depth=result_list[0])
    print('best score is:', result_list[1])
    print('the best max_depth is:', result_list[0])
    return gbdt_model


def choose_best_criterion():
    pass


if __name__ == '__main__':
    x, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
    labels, y = np.unique(y, return_inverse=True)
    grd = GradientBoostingClassifier()
    grd1 = choose_best_loss(grd, 10, 2, x, y, 100, estimation_type="top_num_positive_over_num")
    grd2 = choose_best_learning_rate(grd1, 10, 1, x, y, 100, estimation_type="top_num_positive_over_num", start_num=0, end_num=1, steps=0.1)
    """for grd2.get_params():
    {'criterion': 'friedman_mse','init': None,'learning_rate': 0.6,'loss': 'exponential','max_depth': 3,'max_features': None,
    'max_leaf_nodes': None,'min_impurity_decrease': 0.0,'min_impurity_split': None,'min_samples_leaf': 1,'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,'n_estimators': 100,'presort': 'auto','random_state': None,'subsample': 1.0,'verbose': 0,
    'warm_start': False}
    best score is: 0.9549999999999998
    the best learning_rate is: 0.6
    """
    grd3 = choose_best_n_estimators(grd2, 10, 1, x, y, 100, estimation_type="top_num_positive_over_num", start_num=80, end_num=200, steps=10)
    """for grd3.get_params():
    {'criterion': 'friedman_mse','init': None,'learning_rate': 0.6,'loss': 'exponential','max_depth': 3,'max_features': None,
    'max_leaf_nodes': None,'min_impurity_decrease': 0.0,'min_impurity_split': None,'min_samples_leaf': 1,'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,'n_estimators': 200,'presort': 'auto','random_state': None,'subsample': 1.0,'verbose': 0,
    'warm_start': False}
    best score is: 0.968
    the best n_estimators is: 200
    """
    grd4 = choose_best_max_depth(grd3, 10, 1, x, y, 100, estimation_type="top_num_positive_over_num", start_num=2, end_num=10, steps=1)
    """
    {'criterion': 'friedman_mse','init': None,'learning_rate': 0.6,'loss': 'deviance','max_depth': 6,'max_features': None,
    'max_leaf_nodes': None,'min_impurity_decrease': 0.0,'min_impurity_split': None,'min_samples_leaf': 1,'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,'n_estimators': 200,'presort': 'auto','random_state': None,'subsample': 1.0,'verbose': 0,
    'warm_start': False}
    best score is: 0.958
    the best max_depth is: 6
    """
