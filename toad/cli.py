"""
toad command line application
"""
import sys
import argparse
import pandas as pd


def add_sub(parsers, config):
    """add sub parser by config
    """
    info = config.get('info', {})
    args = config.get('args', [])
    defaults = config.get('defaults', None)

    sub_parser = parsers.add_parser(**info)

    for detail in args:
        flag = detail.pop('flag')
        sub_parser.add_argument(*flag, **detail)

    if defaults:
        sub_parser.set_defaults(**defaults)


def detect(args):
    """detect csv data
    
    Examples:

        toad detect -i xxx.csv -o report.csv
    """
    from .detector import detect

    sys.stdout.write('reading data....\n')
    with args.input as input:
        data = pd.read_csv(input)

    sys.stdout.write('detecting...\n')
    report = detect(data)

    if args.output:
        sys.stdout.write('saving report...\n')
        report.to_csv(args.output)
        sys.stdout.write('report saved!\n')
    else:
        sys.stdout.write(str(report))
        sys.stdout.write('\n')

    return report


DETECT_ARGS = {
    'info': {
        'name': 'detect',
        'description': 'detect data from a csv file',
    },
    'defaults': {
        'func': detect,
    },
    'args': [
        {
            'flag': ('-i', '--input'),
            'type': argparse.FileType(),
            'help': 'the csv file which will be detected',
            'required': True,
        },
        {
            'flag': ('-o', '--output'),
            'type': argparse.FileType('w'),
            'help': 'path of the csv report will be saved',
        },
    ]
}


def get_parser():
    """get parser
    """
    parser = argparse.ArgumentParser(
        prog = 'toad',
        description = 'Detect data from a csv file',
    )

    subparsers = parser.add_subparsers()
    add_sub(subparsers, DETECT_ARGS)

    return parser


def main():
    """
    """
    parser = get_parser()

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
