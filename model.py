# coding: utf-8

import numpy as np
import pandas as pd

from trunk.mytestt.tianchi_1.config import user_id_list


def model1():
    """
    logistic 回归
    :return: 
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    trainSet = pd.read_csv('./DataSet/trainSet.csv')
    # user_id, item_id, type_1_before1, type_2_before1, type_3_before1, continuity_2_days_type1, continuity_2_days_type2, continuity_2_days_type3, labels
    # "continuity_3_days_type1","continuity_3_days_type2", "continuity_3_days_type3",
    trainSet = trainSet[['user_id', 'item_id', "type_1_before1", "type_2_before1", "type_3_before1",
                         "continuity_2_days_type1", "continuity_2_days_type2", "continuity_2_days_type3",
                         "labels"]]

    model.fit(trainSet.ix[:, 2:-1], trainSet.ix[:, -1])
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                       penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                       verbose=0, warm_start=False)

    # 加权逻辑回归（针对类别不平衡，基于代价敏感函数）
    lrW = LogisticRegression(class_weight='balanced')  # 针对样本不均衡问题，设置参数"class_weight
    lrW.fit(trainSet.ix[:, 2:-1], trainSet.ix[:, -1])

    trainLRW_y = lrW.predict(trainSet.ix[:, 2:-1])
    print(trainSet.ix[:, -1].value_counts())
    print(trainLRW_y.sum())

    # 计算精准率等
    from sklearn.cross_validation import cross_val_score
    precisions = cross_val_score(lrW, trainSet.ix[:, 2:-1], trainSet.ix[:, -1], cv=5, scoring='precision')
    print("精确度：\n", np.mean(precisions))

    recalls = cross_val_score(lrW, trainSet.ix[:, 2:-1], trainSet.ix[:, -1], cv=5, scoring='recall')
    print("召回率：\n", np.mean(recalls))

    f1 = cross_val_score(lrW, trainSet.ix[:, 2:-1], trainSet.ix[:, -1], cv=5, scoring='f1')
    print('f1得分：\n', np.mean(f1))

    # 构造输入数据
    dataDay_load = pd.read_csv('./DataSet/dataDay.csv', usecols=['time_day', 'user_id', 'item_id', 'type_1', \
                                                                 'type_2', 'type_3', 'type_4'], index_col='time_day',
                               parse_dates=True)

    predict_x_before3 = dataDay_load.ix['2014-12-16', :]
    predict_x_before3["type_1_before3"] = predict_x_before3["type_1"]
    predict_x_before3["type_2_before3"] = predict_x_before3["type_2"]
    predict_x_before3["type_3_before3"] = predict_x_before3["type_3"]
    predict_x_before3 = predict_x_before3[['user_id', 'item_id', "type_1_before3", "type_2_before3", "type_3_before3"]]

    predict_x_before2 = dataDay_load.ix['2014-12-17', :]
    predict_x_before2["type_1_before2"] = predict_x_before2["type_1"]
    predict_x_before2["type_2_before2"] = predict_x_before2["type_2"]
    predict_x_before2["type_3_before2"] = predict_x_before2["type_3"]
    predict_x_before2 = predict_x_before2[['user_id', 'item_id', "type_1_before2", "type_2_before2", "type_3_before2"]]

    predict_x_before1 = dataDay_load.ix['2014-12-18', :]
    predict_x_before1["type_1_before1"] = predict_x_before1["type_1"]
    predict_x_before1["type_2_before1"] = predict_x_before1["type_2"]
    predict_x_before1["type_3_before1"] = predict_x_before1["type_3"]
    predict_x_before1 = predict_x_before1[['user_id', 'item_id', "type_1_before1", "type_2_before1", "type_3_before1"]]

    predict_x = pd.merge(predict_x_before3, predict_x_before2, on=['user_id', 'item_id'], how='inner').fillna(0.0)
    predict_x = pd.merge(predict_x, predict_x_before1, on=['user_id', 'item_id'], how='inner').fillna(0.0)

    # 用户对同一个Item 操作行为集合
    predict_x['continuity_2_days_type1'] = predict_x["type_1_before1"] + predict_x["type_1_before2"]
    predict_x['continuity_2_days_type2'] = predict_x["type_2_before1"] + predict_x["type_2_before2"]
    predict_x['continuity_2_days_type3'] = predict_x["type_3_before1"] + predict_x["type_3_before2"]

    predict_x['continuity_3_days_type1'] = predict_x["type_1_before1"] + predict_x["type_1_before2"] + predict_x[
        "type_1_before3"]
    predict_x['continuity_3_days_type2'] = predict_x["type_2_before1"] + predict_x["type_2_before2"] + predict_x[
        "type_2_before3"]
    predict_x['continuity_3_days_type3'] = predict_x["type_3_before1"] + predict_x["type_3_before2"] + predict_x[
        "type_3_before3"]

    # 用户在考察日前n天的行为总数计数 用户对同一个Item
    predict_x['u_b_count_in_1'] = predict_x["type_3_before1"] + predict_x["type_2_before1"] + predict_x["type_1_before1"]
    predict_x['u_b_count_in_2'] = predict_x["type_3_before1"] + predict_x["type_2_before1"] + predict_x["type_1_before1"] + predict_x["type_3_before2"] + predict_x["type_2_before2"] + predict_x["type_1_before2"]
    predict_x['u_b_count_in_3'] = predict_x["type_3_before1"] + predict_x["type_2_before1"] + predict_x["type_1_before1"] + predict_x["type_3_before2"] + predict_x["type_2_before2"] + predict_x["type_1_before2"] + predict_x["type_3_before3"] + predict_x["type_2_before3"] + predict_x["type_1_before3"]


    #
    predict_x = predict_x[['user_id', 'item_id', "type_1_before1", "type_2_before1", "type_3_before1",
                           "continuity_2_days_type1", "continuity_2_days_type2", "continuity_2_days_type3"]]

    # 预测集中也剔除30天无购买记录的用户
    print(predict_x.shape)
    predict_x = predict_x[predict_x["user_id"].isin(user_id_list)]
    print(predict_x.shape)

    # 预测 12.19
    predict_y = lrW.predict(predict_x.ix[:, 2:])
    l = predict_y.tolist()
    print(u"count {}".format(l.count(1)))
    print(u"count {}".format(l.count(0)))
    user_item_19 = predict_x.ix[predict_y > 0.0, ['user_id', 'item_id']]  # 选出发生购买行为的用户商品对，即标签为1的，作为最后的提交结果
    user_item_19.to_csv('./DataSet/tianchi_mobile_recommendation_predict.csv', index=False,
                        encoding='utf-8')


def model2():
    """
    GDBT
    :return: 
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import metrics
    gbdt = GradientBoostingClassifier(random_state=10)

    trainSet = pd.read_csv('./DataSet/trainSet.csv')
    X = trainSet.ix[:, 2:6]
    y = trainSet.ix[:, -1]
    gbdt.fit(trainSet.ix[:, 2:6], trainSet.ix[:, -1])
    trainGBDT_y = gbdt.predict(trainSet.ix[:, 2:6])
    print(trainSet.ix[:, -1].value_counts())
    print(trainGBDT_y.sum())
    print("Accuracy : %.4g" % metrics.accuracy_score(trainSet.ix[:, -1].values, trainGBDT_y))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

    # # 调参算出最优n_estimators
    # param_test1 = {'n_estimators': [i for i in range(20, 81, 10)]}
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
    #                                                              min_samples_leaf=20, max_depth=8, max_features='sqrt',
    #                                                              subsample=0.8, random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    # gsearch1.fit(X, y)
    # print (gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    # # 算出最优n_estimators = 20

    # # 对决策树进行调参max_depth
    # param_test2 = {'max_depth': [i for i in range(3, 14, 2)], 'min_samples_split': [i for i in range(100, 801, 200)]}
    # gsearch2 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, min_samples_leaf=20,
    #                                          max_features='sqrt', subsample=0.8, random_state=10),
    #     param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(X, y)
    # print (gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

    # # 调参 min_samples_leaf/min_samples_split
    # param_test3 = {'min_samples_split': [i for i in range(800, 1900, 200)], 'min_samples_leaf': [i for i in range(60, 101, 10)]}
    # gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_depth=5,
    #                                                              max_features='sqrt', subsample=0.8, random_state=10),
    #                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    # gsearch3.fit(X, y)
    # print (gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

    gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_depth=5, min_samples_leaf=70,
                                      min_samples_split=800, max_features='sqrt', subsample=0.8, random_state=10)
    gbm1.fit(X, y)
    y_pred = gbm1.predict(X)
    y_predprob = gbm1.predict_proba(X)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

    # # 计算精准率等
    # from sklearn.cross_validation import train_test_split, cross_val_score
    # precisions = cross_val_score(gbdt, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='precision')
    # print ("精确度：\n", np.mean(precisions))
    #
    # recalls = cross_val_score(gbdt, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='recall')
    # print ("召回率：\n", np.mean(recalls))
    #
    # f1 = cross_val_score(gbdt, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='f1')
    # print ('f1得分：\n', np.mean(f1))

    # 构造输入数据
    dataDay_load = pd.read_csv('./DataSet/dataDay.csv', usecols=['time_day', 'user_id', 'item_id', 'type_1', \
                                                                 'type_2', 'type_3', 'type_4'], index_col='time_day',
                               parse_dates=True)
    predict_x = dataDay_load.ix['2014-12-18', :]

    print(predict_x.head(20))

    # 预测
    predict_y = gbdt.predict(predict_x.ix[:, 2:])
    l = predict_y.tolist()
    print(u"count {}".format(l.count(1)))
    print(u"count {}".format(l.count(0)))
    # predict_y = pd.DataFrame(predict_y, columns=["predict_y"])
    # print (predict_y.info())
    # print(predict_x.info())
    # predict_x = pd.concat([predict_x, predict_y], axis = 1)
    # print (type(predict_y))
    # print (type(predict_x))
    # print (predict_x)
    # print (predict_y)
    user_item_19 = predict_x.ix[predict_y > 0.0, ['user_id', 'item_id']]  # 选出发生购买行为的用户商品对，即标签为1的，作为最后的提交结果
    # print (user_item_19)
    # print (type(user_item_19))
    user_item_19.to_csv('./DataSet/tianchi_mobile_recommendation_predict.csv', index=False,
                        encoding='utf-8')


def model3():
    """
    随机森林
    :return: 
    """
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()

    trainSet = pd.read_csv('./DataSet/trainSet.csv')
    rf.fit(trainSet.ix[:, 2:6], trainSet.ix[:, -1])

    trainRF_y = rf.predict(trainSet.ix[:, 2:6])
    print(trainSet.ix[:, -1].value_counts())
    print(trainRF_y.sum())

    # 计算精准率等
    from sklearn.cross_validation import cross_val_score
    precisions = cross_val_score(rf, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='precision')
    print("精确度：\n", np.mean(precisions))

    recalls = cross_val_score(rf, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='recall')
    print("召回率：\n", np.mean(recalls))

    f1 = cross_val_score(rf, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='f1')
    print('f1得分：\n', np.mean(f1))

    # 构造输入数据
    dataDay_load = pd.read_csv('./DataSet/dataDay.csv', usecols=['time_day', 'user_id', 'item_id', 'type_1', \
                                                                 'type_2', 'type_3', 'type_4'], index_col='time_day',
                               parse_dates=True)
    predict_x = dataDay_load.ix['2014-12-18', :]

    print(predict_x.head(20))

    # 预测
    predict_y = rf.predict(predict_x.ix[:, 2:])
    l = predict_y.tolist()
    print(u"count {}".format(l.count(1)))
    # predict_y = pd.DataFrame(predict_y, columns=["predict_y"])
    # print (predict_y.info())
    # print(predict_x.info())
    # predict_x = pd.concat([predict_x, predict_y], axis = 1)
    # print (type(predict_y))
    # print (type(predict_x))
    # print (predict_x)
    # print (predict_y)
    user_item_19 = predict_x.ix[predict_y > 0.0, ['user_id', 'item_id']]  # 选出发生购买行为的用户商品对，即标签为1的，作为最后的提交结果
    # print (user_item_19)
    # print (type(user_item_19))
    user_item_19.to_csv('./DataSet/tianchi_mobile_recommendation_predict.csv', index=False,
                        encoding='utf-8')


def model4():
    """
    lightgbm
    :return: 
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 创建成lgb特征的数据集格式
    trainSet = pd.read_csv('./DataSet/trainSet.csv')
    X_train, X_test, y_train, y_test = train_test_split(trainSet.ix[:, 2:6], trainSet.ix[:, -1], test_size=0.2)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    # 预测数据集
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    print(X_test.sum())
    print(y_pred.sum())

    # # 计算精准率等
    # from sklearn.cross_validation import train_test_split, cross_val_score
    # precisions = cross_val_score(rf, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='precision')
    # print ("精确度：\n", np.mean(precisions))
    #
    # recalls = cross_val_score(rf, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='recall')
    # print ("召回率：\n", np.mean(recalls))
    #
    # f1 = cross_val_score(rf, trainSet.ix[:, 2:6], trainSet.ix[:, -1], cv=5, scoring='f1')
    # print ('f1得分：\n', np.mean(f1))

    # 构造输入数据
    dataDay_load = pd.read_csv('./DataSet/dataDay.csv', usecols=['time_day', 'user_id', 'item_id', 'type_1', \
                                                                 'type_2', 'type_3', 'type_4'], index_col='time_day',
                               parse_dates=True)
    predict_x = dataDay_load.ix['2014-12-18', :]
    predict_y = gbm.predict(predict_x, num_iteration=gbm.best_iteration)
    print(u"count {}".format(predict_y.sum()))

    user_item_19 = predict_x.ix[predict_y > 0.0, ['user_id', 'item_id']]  # 选出发生购买行为的用户商品对，即标签为1的，作为最后的提交结果
    # print (user_item_19)
    # print (type(user_item_19))
    user_item_19.to_csv('./DataSet/tianchi_mobile_recommendation_predict.csv', index=False,
                        encoding='utf-8')


if "__main__" == __name__:
    # preData()
    # feature()
    # trainDate()
    model1()
