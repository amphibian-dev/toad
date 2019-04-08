import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import xlsxwriter
import toad
from toad.transform import WOETransformer





# 把index数字转换成区间
def rename_columns(index_list, bins, is_float=False):
    list = [int(x) for x in index_list]
    bins_list = []
    left = '[0,'
    if -9999999 in list:
        list.remove(-9999999)
        bins_list.append('[-9999999,-9999999]')
        left = '(-9999999,'
    for i in range(len(list)):
        if i == 0:
            bins_list.append(left + str(float('%.2f' % bins[list[0]]) if is_float else int(bins[list[0]])) + ')')
        elif i == len(list) - 1:
            bins_list.append('[' + str(float('%.2f' % bins[list[len(list) - 1] - 1]) if is_float else int(
                bins[list[len(list) - 1] - 1])) + ',inf)')
        else:
            bins_list.append(
                '[' + str(float('%.2f' % bins[list[i - 1]]) if is_float else int(bins[list[i - 1]])) + ',' + str(
                    float('%.2f' % bins[list[i]]) if is_float else int(bins[list[i]])) + ')')
    return bins_list


# 离散变量分组求坏账率
def self_bin_dispersed(X, Y):
    d1 = pd.DataFrame({"X": X, "Y": Y})
    d2 = d1.groupby("X")
    d3 = pd.DataFrame(columns=['bins', 'bad', 'fre', 'bad_rate'])
    for name, group in d2:
        bad_count = group.sum().Y
        total = group.count().Y
        bad_rate = group.mean().Y
        d3 = d3.append({'bins': name, 'bad': bad_count, 'fre': total, 'bad_rate': bad_rate}, ignore_index=True)
    d4 = d3.sort_values(by='bins')
    return d4


# 连续变量分组求坏账率
def self_bin_successive(X, Y, bins):
    d1 = pd.DataFrame({"X": X, "Y": Y, 'index': X})
    d2 = d1.groupby('index')
    d3 = pd.DataFrame()
    d3['index'] = d2.min().X
    d3['bad'] = d2.sum().Y
    d3['fre'] = d2.count().Y
    d3['bad_rate'] = d2.mean().Y
    d3.sort_values(by='index', inplace=True)
    index = d3['index'].tolist()
    bins_list = rename_columns(index, bins, True)
    d3['bins'] = bins_list
    d3.drop('index', axis=1, inplace=True)
    return d3


def draw_data(workbook,worksheet,data,**kwargs):
    start_index = kwargs['start_index']
    title = kwargs['title']
    columns_list =kwargs['columns']
    haed_format = workbook.add_format({'bold': True, 'valign': 'middle', 'align': 'center', 'border': 1})
    data_format = workbook.add_format({'valign': 'middle', 'align': 'center', 'border': 1})
    merge_format = workbook.add_format({'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    worksheet.merge_range(
        "A" + str(start_index + 1) + ":" + column_to_name(len(columns_list) - 1 ).upper() + str(start_index + 1),
        title, merge_format)
    worksheet.write_row('A' + str(start_index + 2), columns_list, haed_format)
    for j in range(len(columns_list)):
        worksheet.write_column(column_to_name(j) + str(start_index + 3), data[columns_list[j]], data_format)
    return len(data)+ 4 + start_index
# 单个变量的分组结果和图写入到excel
def save_bins_and_chart(worksheet, workbook, data, var_name, start_index):
    draw_data(workbook,worksheet,data,draw_index=False,start_index=start_index,title=var_name, columns = ['bins', 'bad', 'fre', 'bad_rate'])
    column_chart1 = workbook.add_chart({'type': 'column'})
    column_chart1.add_series({
        'name': '=分布!$B$' + str(start_index + 2),
        'categories': '=分布!$A$' + str(start_index + 3) + ':$A$' + str(start_index + 2 + len(data)),
        'values': '=分布!$B$' + str(start_index + 3) + ':$B$' + str(start_index + 2 + len(data)),

    })
    column_chart1.set_size({'width': 700, 'height': 380})
    column_chart1.add_series({
        'name': '=分布!$C$' + str(start_index + 2),
        'categories': '=分布!$A$' + str(start_index + 3) + ':$A$' + str(start_index + 2 + len(data)),
        'values': '=分布!$C$' + str(start_index + 3) + ':$C$' + str(start_index + 2 + len(data)),
    })
    line_chart1 = workbook.add_chart({'type': 'line'})
    line_chart1.add_series({
        'name': '=分布!$D$' + str(start_index + 2),
        'categories': '=分布!$A$' + str(start_index + 3) + ':$A$' + str(start_index + 2 + len(data)),
        'values': '=分布!$D$' + str(start_index + 3) + ':$D$' + str(start_index + 2 + len(data)),
        'y2_axis': True,
    })
    column_chart1.combine(line_chart1)
    column_chart1.set_title({'name': var_name})
    column_chart1.set_x_axis({'name': '分组'})
    column_chart1.set_y_axis({'name': '个数'})
    column_chart1.set_y2_axis({'name': '坏账率'})
    worksheet.insert_chart('F' + str(start_index + 1), column_chart1)
    return max(22 + start_index, start_index + len(data) + 4)


# 选择iv大于0.02或者iv前十的变量
def select_iv(quality,num,iv_threshold_value):
    if len(quality[quality['iv'] > 0]) < num:
        return quality[quality['iv'] > 0]
    else:
        quality.sort_values(by="iv", ascending=False)
        high_iv = quality[quality['iv'] > iv_threshold_value]
        return high_iv if len(high_iv) >= num else quality[0:num]


# 合并长尾数据并进行等步长分组
def merger_data(data, var, unique_num,is_merge_high=True):
    if is_numeric_dtype(data[var]) and data[var].nunique() > unique_num:
        data_miss = data[data[var] == -9999999]
        data_nomiss = data[data[var] != -9999999]
        merge_high_data = data_nomiss[var]
        if is_merge_high:
            merge_high_data = toad.utils.clip(data_nomiss[var], quantile=(None, .99))
        data_index, bins = toad.merge(merge_high_data, method='step', return_splits=True, n_bins=unique_num)
        temp = pd.DataFrame(data_index, columns=[var])
        temp = temp.append(data_miss[[var]], ignore_index=True)[var]
        target = data_nomiss.append(data_miss, ignore_index=True)['target']
        return temp, target, bins
    else:
        return data[var], None, None


# 两两指标做透视表
def crosstab_data(columns_var, row_var, data,unique_num,*args):
    columns_data, columns_target, columns_bins = merger_data(data, columns_var, unique_num,args[0])
    row_data, row_target, row_bins = merger_data(data, row_var, unique_num,args[1])
    result = pd.crosstab(row_data, columns_data, margins=True, dropna=False)
    if columns_bins is not None:
        columns = result.columns.tolist()
        columns.remove('All')
        columns_bins_list = rename_columns(columns, columns_bins, args[2])
        columns_bins_list.append('All')
        result.set_axis(columns_bins_list, axis=1, inplace=True)
    if row_bins is not None:
        index = result.index.tolist()
        index.remove('All')
        index_bins_list = rename_columns(index, row_bins, args[3])
        index_bins_list.append('All')
        result.set_axis(index_bins_list, axis=0, inplace=True)
    return result


# 写入所有高iv的变量分组和图到excel
def all_var_pic(high_iv_var_list, data, workbook, worksheet,start_index,unique_num):
    for i in range(len(high_iv_var_list)):
        var_name = high_iv_var_list[i]
        result = None
        data_index, target, bins = merger_data(data, var_name, unique_num,True)
        if bins is not None:
            result = self_bin_successive(data_index, target, bins)
        else:
            result = self_bin_dispersed(data[var_name], data['target'])
        start_index = save_bins_and_chart(worksheet, workbook, result, var_name, start_index)
    return start_index


# 写入所有指标的逾期天数透视表到excel
def save_overdue_crosstab(data, var_list, workbook, worksheet,start_index,unique_num):
    for i in range(len(var_list)):
        var = var_list[i]
        result = crosstab_data(var, 'overdue_days', data,unique_num,True, False, True, False)
        columns = ['overdue_days'] + result.columns.tolist()
        result['overdue_days'] = result.index
        start_index = draw_data(workbook,worksheet,result,start_index=start_index,title=var,columns=columns)
    return start_index


# 进入模型的变量分组，少于10个的一个一组，大于10个的,按照10%分为一组
def var_bins(quality):
    quality.sort_values(by='iv', ascending=False, inplace=True)
    var_group_list = []
    if len(quality) < 10:
        for temp in quality.index.tolist():
            var_group_list.append([temp])
    else:
        bins = pd.qcut(range(len(quality)), 10, labels=False)
        df_var = pd.DataFrame(columns=['num', 'var', 'iv'])
        df_var['num'] = bins
        df_var['var'] = quality.index
        for group, temp in df_var.groupby(by='num'):
            var_group_list.append(temp['var'].tolist())
    return var_group_list


# 用woe替换离散变量
def replace_with_woe(train_data, test_data, exclude_var, target='target'):
    all_var = train_data.columns.tolist()
    for var in all_var:
        if var not in exclude_var and is_object_dtype(train_data[var]):
            woe = WOETransformer().fit(train_data[var], train_data[target])
            train_data[var] = woe.transform(train_data[var])
            train_data[var].astype('float64')
            test_data[var]  = woe.transform(test_data[var])
            test_data[var].astype('float64')
    return train_data, test_data


# 按照变量分组循环建模，并计算ks
def compute_ks(data_train, data_test, quality, self_var_list, self_var_ks, result, target='target'):
    test_var_list = var_bins(quality)
    best_ks = self_var_ks
    best_input_var = []
    for temp in test_var_list:
        input_var = best_input_var + temp
        model, param = param_opt(data_train[input_var + self_var_list], data_train[target])
        train_ks = get_ks(model, data_train[input_var + self_var_list], data_train[target])
        test_ks = get_ks(model, data_test[input_var + self_var_list], data_test[target])
        if best_ks < test_ks:
            best_ks = test_ks
            best_input_var = best_input_var + input_var
        result = result.append({'train_ks': train_ks, 'test_ks': test_ks, 'var': str(input_var)}, ignore_index=True)
    return result


# 把序列号转成excel列名
def column_to_name(index):
    if index > 25:
        ch1 = chr(index % 26 + 65)
        ch2 = chr(int(index / 26) + 64)
        return ch2 + ch1
    else:
        return chr(index % 26 + 65)


def ks_score(y, y_pred):
    return toad.KS( y_pred[:,1],y)

scorer = make_scorer(ks_score, needs_proba = True)

def test_score(p, x_train, y_train):
    model_ada = AdaBoostClassifier(
        algorithm='SAMME.R',
        base_estimator=None,
        learning_rate=0.1,
        n_estimators=p,
        random_state=123,
    )
    score_mean = cross_val_score(model_ada, x_train, y_train, cv=10, scoring = scorer).mean()
    return score_mean

def findMax(start, end, results, train_x, train_y):
    if end - start <= 10:
        if results[start] > results[end]:
            return start, results[start]
        else:
            return end, results[end]
    need_test = int((start + end) / 2)
    results.at[need_test] = test_score(need_test, train_x, train_y)
    if results[need_test] > max(results[start], results[end]) or results[need_test] < min(results[start], results[end]):
        p1, r1 = findMax(start, need_test, results, train_x, train_y)
        p2, r2 = findMax(need_test, end, results, train_x, train_y)
        if r1 > r2:
            return p1, r1
        else:
            return p2, r2
    elif results[start] > results[end]:
        return findMax(start, need_test, results, train_x, train_y)
    else:
        return findMax(need_test, end, results, train_x, train_y)
def param_opt(train_x, train_y, start = 100, end = 700):
    results = pd.Series()
    results.at[start] = test_score(start, train_x, train_y)
    results.at[end] = test_score(end, train_x, train_y)
    param, mean = findMax(start, end, results, train_x, train_y)
    model = AdaBoostClassifier(
        algorithm='SAMME.R',
        learning_rate=0.1,
        n_estimators=param,
        random_state=123,
    )
    model.fit(train_x, train_y)
    return model,param

def get_ks(model, X, Y):
    pre = model.predict_proba(X)
    return ks_score(Y, pre)



def evaluate(test_data, excel_name = 'report.xlsx', num = 10, iv_threshold_value = 0.02, unique_num = 20, overdue_days = False, self_data = None):
    # 测试数据iv等信息并写入excel
    workbook = xlsxwriter.Workbook(excel_name)
    quality = toad.quality(test_data.drop(columns=["loan_apply_no"]), target="target")
    quality.sort_values(by='iv',ascending=False,inplace=True)
    quality['var_name'] = quality.index
    draw_data(workbook, workbook.add_worksheet('quality'), quality, start_index=0, title='变量探查结果',
              columns=['var_name', 'iv', 'gini', 'entropy', 'unique'])
    print("quality 计算完毕")
    # 选择高iv的测试变量分组并写入excel
    quality = quality.replace("--", -1)
    high_iv_var = select_iv(quality,num,iv_threshold_value)
    high_iv_var_list = high_iv_var.index.tolist()
    all_var_pic(high_iv_var_list, test_data, workbook, workbook.add_worksheet("分布"), 0,unique_num)
    print("变量分组处理完毕")
    all_data = None
    if overdue_days or self_data is not None:
        all_data = pd.merge(self_data, test_data.drop(columns=['target']), how='left', on='loan_apply_no')
    if overdue_days:
        # 数据合并
        test_var = test_data.columns.tolist()
        test_var.remove("loan_apply_no")
        test_var.remove("target")
        save_overdue_crosstab(all_data, test_var, workbook, workbook.add_worksheet("逾期天数"), 0,unique_num=unique_num)
        print("逾期天数处理完毕")
    if self_data is not None:
        # train和test数据集划分
        data_train = all_data[all_data['SMP_N'] == 'dev_smp']
        data_test = all_data[all_data['SMP_N'] == 'ver_smp']
        # 离散变量做woe处理
        data_train, data_test = replace_with_woe(data_train, data_test, exclude_var=['SMP_N', 'loan_apply_no'],
                                                 target='target')
        # 循环加入变量建模并计算模型的ks最后写入excel
        self_data_var_list = self_data.columns.tolist()
        drop_var = ['loan_apply_no', 'SMP_N', 'target']
        for var in drop_var:
            self_data_var_list.remove(var)
        if 'overdue_days' in self_data_var_list:
            self_data_var_list.remove('overdue_days')
        model, param = param_opt(data_train[self_data_var_list], data_train['target'])
        train_ks = get_ks(model, data_train[self_data_var_list], data_train['target'])
        test_ks = get_ks(model, data_test[self_data_var_list], data_test['target'])
        result = pd.DataFrame(columns=['train_ks', 'test_ks', 'var'])
        result = result.append({'train_ks': train_ks, 'test_ks': test_ks, 'var': '[]'}, ignore_index=True)
        result = compute_ks(data_train, data_test, quality, self_data_var_list, test_ks, result, target='target')
        draw_data(workbook, workbook.add_worksheet("ks"), result, start_index=0, title='ks测试结果',
                  columns=result.columns.tolist())
        print("模型计算完毕")
    workbook.close()
