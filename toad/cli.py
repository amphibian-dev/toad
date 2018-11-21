"""
toad command line application
"""
import pkgutil
import os
import sys
import argparse
import pandas as pd
from importlib import import_module

COMMANDS = 'commands'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMAND_DIR = os.path.join(CURRENT_DIR, COMMANDS)

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


def get_plugins():
    plugins = []

    for _, name, ispkg in pkgutil.iter_modules([COMMAND_DIR]):
        if ispkg:
            module = import_module('toad.{}.{}'.format(COMMANDS, name))
            plugins.append(module)

    return plugins


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
