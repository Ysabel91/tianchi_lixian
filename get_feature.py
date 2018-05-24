# coding: utf-8

import pandas as pd


def preData():
    userAll = pd.read_csv('./DataSet/tianchi_fresh_comp_train_user.csv', \
                          usecols=['user_id', 'item_id', 'behavior_type', 'time'])

    print(userAll.duplicated().sum())

    itemSub = pd.read_csv('./DataSet/tianchi_fresh_comp_train_item.csv', usecols=['item_id'])

    print(itemSub.item_id.is_unique)

    print(itemSub.item_id.value_counts().head())

    itemSet = itemSub[['item_id']].drop_duplicates()  # 去除重复的行

    userSub = pd.merge(userAll, itemSet, on='item_id', how='inner')
    print(userSub.info())
    print(userSub.head())

    userSub.to_csv('./DataSet/userSub.csv')


def feature():
    userSub = pd.read_csv('./DataSet/userSub.csv', usecols=['user_id', 'item_id', 'behavior_type', 'time'],
                          parse_dates=True)
    # userSub = userSub.set_index('time').sort_index()

    typeDummies = pd.get_dummies(userSub['behavior_type'], prefix='type')  # onehot哑变量编码
    # print (typeDummies.head())

    userSubOneHot = pd.concat([userSub[['user_id', 'item_id', 'time']], typeDummies], axis=1)

    usertem = pd.concat([userSub[['user_id', 'item_id']], typeDummies, userSub[['time']]], axis=1)  # 将哑变量特征加入数据表中

    userSubOneHotGroup = userSubOneHot.groupby(['time', 'user_id', 'item_id'],
                                               as_index=False).sum()  # 另外一种方法是在sum（）后使用.reset_index()

    # 剔除30天内无购买记录用户
    userBuyTable = usertem[["user_id", "type_4"]]
    userBuyTable = userBuyTable.groupby(['user_id'], as_index=False).sum()
    userBuyTable = userBuyTable[userBuyTable["type_4"] > 0]
    user_id_list = list(userBuyTable["user_id"])
    userSubOneHotGroup = userSubOneHotGroup[userSubOneHotGroup["user_id"].isin(user_id_list)]

    # 拆分天和小时
    userSubOneHotGroup['time_day'] = pd.to_datetime(userSubOneHotGroup.time.values).date

    userSubOneHotGroup['time_hour'] = pd.to_datetime(userSubOneHotGroup.time.values).time

    print(userSubOneHotGroup.head())

    dataHour = userSubOneHotGroup.ix[:, 0:7]
    dataHour.to_csv('./DataSet/dataHour.csv')

    dataDay = userSubOneHotGroup.groupby(['time_day', 'user_id', 'item_id'], as_index=False).sum()
    dataDay.to_csv('./DataSet/dataDay.csv')

def trainDate():
    dataDay_load = pd.read_csv('./DataSet/dataDay.csv', usecols=['time_day', 'user_id', 'item_id', 'type_1', \
                                                                 'type_2', 'type_3', 'type_4'], index_col='time_day',
                               parse_dates=True)

    train_x_before3 = dataDay_load.ix['2014-12-14', :]  # 14号选取特征数据集
    train_x_before3["type_1_before3"] = train_x_before3["type_1"]
    train_x_before3["type_2_before3"] = train_x_before3["type_2"]
    train_x_before3["type_3_before3"] = train_x_before3["type_3"]
    train_x_before3 = train_x_before3[['user_id', 'item_id', "type_1_before3", "type_2_before3", "type_3_before3"]]

    train_x_before2 = dataDay_load.ix['2014-12-15', :]  # 15号选取特征数据集
    train_x_before2["type_1_before2"] = train_x_before2["type_1"]
    train_x_before2["type_2_before2"] = train_x_before2["type_2"]
    train_x_before2["type_3_before2"] = train_x_before2["type_3"]
    train_x_before2 = train_x_before2[['user_id', 'item_id', "type_1_before2", "type_2_before2", "type_3_before2"]]

    train_x_before1 = dataDay_load.ix['2014-12-16', :]  # 16号选取特征数据集
    train_x_before1["type_1_before1"] = train_x_before1["type_1"]
    train_x_before1["type_2_before1"] = train_x_before1["type_2"]
    train_x_before1["type_3_before1"] = train_x_before1["type_3"]
    train_x_before1 = train_x_before1[['user_id', 'item_id', "type_1_before1", "type_2_before1", "type_3_before1"]]

    train_x = pd.merge(train_x_before3, train_x_before2, on=['user_id', 'item_id'], how='inner').fillna(0.0)
    train_x = pd.merge(train_x, train_x_before1, on=['user_id', 'item_id'], how='inner').fillna(0.0)
    # print(train_x.head())

    train_x['continuity_2_days_type1'] = train_x["type_1_before1"] + train_x["type_1_before2"]
    train_x['continuity_2_days_type2'] = train_x["type_2_before1"] + train_x["type_2_before2"]
    train_x['continuity_2_days_type3'] = train_x["type_3_before1"] + train_x["type_3_before2"]

    train_x['continuity_3_days_type1'] = train_x["type_1_before1"] + train_x["type_1_before2"] + train_x[
        "type_1_before3"]
    train_x['continuity_3_days_type2'] = train_x["type_2_before1"] + train_x["type_2_before2"] + train_x[
        "type_2_before3"]
    train_x['continuity_3_days_type3'] = train_x["type_3_before1"] + train_x["type_3_before2"] + train_x[
        "type_3_before3"]

    print(train_x.head())

    # # 用户在考察日前n天的行为总数计数 用户对同一个Item
    # predict_x['u_b_count_in_1'] = predict_x["type_3_before1"] + predict_x["type_2_before1"] + predict_x[
    #     "type_1_before1"]
    # predict_x['u_b_count_in_2'] = predict_x["type_3_before1"] + predict_x["type_2_before1"] + predict_x[
    #     "type_1_before1"] + predict_x["type_3_before2"] + predict_x["type_2_before2"] + predict_x["type_1_before2"]
    # predict_x['u_b_count_in_3'] = predict_x["type_3_before1"] + predict_x["type_2_before1"] + predict_x[
    #     "type_1_before1"] + predict_x["type_3_before2"] + predict_x["type_2_before2"] + predict_x["type_1_before2"] + \
    #                               predict_x["type_3_before3"] + predict_x["type_2_before3"] + predict_x[
    #                                   "type_1_before3"]

    train_y = dataDay_load.ix['2014-12-17', ['user_id', 'item_id', 'type_4']]  # 17号的购买行为作为分类标签
    dataSet = pd.merge(train_x, train_y, on=['user_id', 'item_id'], suffixes=('_x', '_y'), how='left').fillna(
        0.0)  # 特征数据和标签数据构成训练数据集

    dataSet['labels'] = dataSet.type_4.map(lambda x: 1.0 if x > 0.0 else 0.0)
    dataSet.drop(
        ["type_1_before2", "type_2_before2", "type_3_before2", "type_1_before3", "type_2_before3", "type_3_before3",
         "type_4"], axis=1, inplace=True)

    trainSet = dataSet.copy()  # 重命名并保存训练数据集
    trainSet.to_csv('./DataSet/trainSet.csv')















    test_x_before3 = dataDay_load.ix['2014-12-15', :]  # 15号选取特征数据集
    test_x_before3["type_1_before3"] = test_x_before3["type_1"]
    test_x_before3["type_2_before3"] = test_x_before3["type_2"]
    test_x_before3["type_3_before3"] = test_x_before3["type_3"]
    test_x_before3 = test_x_before3[['user_id', 'item_id', "type_1_before3", "type_2_before3", "type_3_before3"]]

    test_x_before2 = dataDay_load.ix['2014-12-16', :]  # 16号特征数据集，最为测试输入数据集
    test_x_before2["type_1_before2"] = test_x_before2["type_1"]
    test_x_before2["type_2_before2"] = test_x_before2["type_2"]
    test_x_before2["type_3_before2"] = test_x_before2["type_3"]
    test_x_before2 = test_x_before2[['user_id', 'item_id', "type_1_before2", "type_2_before2", "type_3_before2"]]

    test_x_before1 = dataDay_load.ix['2014-12-17', :]  # 17号特征数据集，最为测试输入数据集
    test_x_before1["type_1_before1"] = test_x_before1["type_1"]
    test_x_before1["type_2_before1"] = test_x_before1["type_2"]
    test_x_before1["type_3_before1"] = test_x_before1["type_3"]
    test_x_before1 = test_x_before1[['user_id', 'item_id', "type_1_before1", "type_2_before1", "type_3_before1"]]

    test_x = pd.merge(test_x_before3, test_x_before2, on=['user_id', 'item_id'], how='inner').fillna(0.0)
    test_x = pd.merge(test_x, test_x_before1, on=['user_id', 'item_id'], how='inner').fillna(0.0)

    test_x['continuity_2_days_type1'] = test_x["type_1_before1"] + test_x["type_1_before2"]
    test_x['continuity_2_days_type2'] = test_x["type_2_before1"] + test_x["type_2_before2"]
    test_x['continuity_2_days_type3'] = test_x["type_3_before1"] + test_x["type_3_before2"]

    test_x['continuity_3_days_type1'] = test_x["type_1_before1"] + test_x["type_1_before2"] + test_x["type_1_before3"]
    test_x['continuity_3_days_type2'] = test_x["type_2_before1"] + test_x["type_2_before2"] + test_x["type_2_before3"]
    test_x['continuity_3_days_type3'] = test_x["type_3_before1"] + test_x["type_3_before2"] + test_x["type_3_before3"]

    test_y = dataDay_load.ix['2014-12-18', ['user_id', 'item_id', 'type_4']]  # 18号购买行为作为测试标签数据集

    testSet = pd.merge(test_x, test_y, on=['user_id', 'item_id'], suffixes=('_x', '_y'), how='left').fillna(
        0.0)  # 构成测试数据集
    testSet['labels'] = testSet.type_4.map(lambda x: 1.0 if x > 0.0 else 0.0)
    testSet.drop(
        ["type_1_before2", "type_2_before2", "type_3_before2", "type_1_before3", "type_2_before3", "type_3_before3",
         "type_4"], axis=1, inplace=True)

    testSet.to_csv('./DataSet/testSet.csv')
