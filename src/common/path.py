import os


def check_dir(*args):
    """
    Check argument(s) is directory.
    If not, generate directory.
    """
    for a in args:
        assert isinstance(a, str)
        if a.endswith('/'): a = a[:-1]
        if not os.path.isdir(a):
            check_dir(os.path.split(a)[0])
            os.mkdir(a)


def path_finder(root: str, extension: str):
    """
    Finds all files with a given extension in all subdirectories.
    """
    filenames = []
    bag = os.listdir(root)
    for b in bag:
        new_path = os.path.join(root, b)
        if os.path.isdir(new_path):
            filenames += path_finder(new_path, extension)
        else:
            if b.endswith(extension):
                filenames.append(new_path)
            else:
                continue
    return filenames
