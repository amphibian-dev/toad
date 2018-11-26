import os
import pkgutil
from importlib import import_module

COMMAND_DIR = os.path.dirname(os.path.abspath(__file__))

def get_plugins():
    plugins = []

    for _, name, ispkg in pkgutil.iter_modules([COMMAND_DIR]):
        if ispkg:
            module = import_module('toad.commands.{}'.format(name))
            plugins.append(module)

    return plugins
