__version_info__ = (0, 0, 62, 'final', 0)

def get_version(version):
    main = '.'.join(str(x) for x in version[:3])

    if version[3] == 'final':
        return main

    symbol = {
        'alpha': 'a',
        'beta': 'b',
        'rc': 'rc',
    }

    return main + symbol[version[3]] + str(version[4])

__version__ = get_version(__version_info__)
