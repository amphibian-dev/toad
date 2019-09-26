import sys
import argparse
import pandas as pd

def func(args):
    """detect csv data

    Examples:

        toad evaluate -i xxx.csv
    """
    from .evaluate import evaluate


    sys.stdout.write('reading data....\n')

    test_data = pd.read_csv(args.input)
    if args.base is not None:
        self_data = pd.read_csv(args.base)
    else:
        self_data = None

    arguments = {
        'excel_name': args.name,
        'num': args.top,
        'iv_threshold_value': args.iv,
        'unique_num': args.unique,
        'self_data': self_data,
        'overdue_days': args.overdue,
    }

    evaluate(test_data, **arguments)


ARGS = {
    'info': {
        'name': 'evaluate',
        'description': '第三方数据评估',
    },
    'defaults': {
        'func': func,
    },
    'args': [
        {
            'flag': ('-i', '--input'),
            'type': argparse.FileType('r', encoding='utf-8'),
            'help': '需要评估的 csv 文件',
            'required': True,
        },
        {
            'flag': ('--base',),
            'type': argparse.FileType('r', encoding='utf-8'),
            'help': '用于测试提升效果的基准 csv 数据文件',
            'default': None,
        },
        {
            'flag': ('--overdue',),
            'help': '是否启用逾期天数分析',
            'action': 'store_true',
        },
        {
            'flag': ('--top',),
            'type': int,
            'help': '选择 IV 最高的 n 个变量分析',
            'default': 10,
        },
        {
            'flag': ('--iv',),
            'type': float,
            'help': '选择 IV 大于阈值的变量进行分析',
            'default': 0.02,
        },
        {
            'flag': ('--unique',),
            'type': int,
            'help': '将连续变量合并成 n 组进行分析',
            'default': 10,
        },
        {
            'flag': ('--name',),
            'type': str,
            'help': '生成报告的文件名',
            'default': 'report.xlsx',
        },
    ]
}
