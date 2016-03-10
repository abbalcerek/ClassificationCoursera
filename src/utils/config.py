
def go_down_dir(path, levels):
    if levels and levels <= 0:
        return path
    import os
    return go_down_dir(os.path.dirname(path), levels - 1)


def project_root():
    import os
    curr_path = os.path.realpath(__file__)
    return go_down_dir(curr_path, 2)
