import os
import copy
import h5py


def check_end_character(string, character):
    if string != '':
        if string[-1] != character:
            string = f'{string}{character}'
    return string

def check_underscore(string):
    """Assure string ends with an underscore

    Parameter
    ---------
    string:str

    Returns
    -------
    string:str
    """
    return check_end_character(string, "_")

def check_slash(string):
    """Assure string ends with a slash

    Parameter
    ---------
    string:str

    Returns
    -------
    string:str
    """
    return check_end_character(string, "/")

def check_folder_exists_or_create(folder, return_folder=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if return_folder:
        return check_slash(folder)

def h5_tree(val:h5py.File, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))