
def go_down_dir(path, levels):
    if levels and levels <= 0:
        return path
    import os
    return go_down_dir(os.path.dirname(path), levels - 1)


def project_root(path=''):
    import os
    curr_path = os.path.realpath(__file__)
    return "\\".join([go_down_dir(curr_path, 2), path])


def pandas_setup():
    import pandas as pd
    pd.set_option('expand_frame_repr', False)
    pd.options.mode.chained_assignment = None