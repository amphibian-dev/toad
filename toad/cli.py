"""
toad command line application
"""
import argparse
from .commands import get_plugins


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


def get_parser():
    """get parser
    """
    parser = argparse.ArgumentParser(
        prog = 'toad',
        description = 'Detect data from a csv file',
    )

    subparsers = parser.add_subparsers()

    plugins = get_plugins()
    for plug in plugins:
        add_sub(subparsers, plug.ARGS)

    return parser


def main():
    """
    """
    parser = get_parser()

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
