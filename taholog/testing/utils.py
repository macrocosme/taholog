import os
import h5py


def check_end_character(string, character):
    """
    Checks if the given string ends with the specified character.

    Args:
        string (str): The input string to check.
        character (str): The character to check for at the end of the string.

    Returns:
        str: The modified string with the character appended at the end, if necessary.
    """
    if string != "":
        if string[-1] != character:
            string = f"{string}{character}"
    return string


def check_underscore(string):
    """
    Check if the given string ends with an underscore.

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string ends with an underscore, False otherwise.
    """
    return check_end_character(string, "_")


def check_slash(string):
    """
    Checks if the given string ends with a slash ("/").

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string ends with a slash, False otherwise.
    """
    return check_end_character(string, "/")


def check_folder_exists_or_create(folder, return_folder=True):
    """
    Checks if a folder exists, and creates it if it doesn't.

    Args:
        folder (str): The path of the folder to check/create.
        return_folder (bool, optional): Whether to return the folder path after checking/creating.
                                        Defaults to True.

    Returns:
        str: The folder path, with a trailing slash if `return_folder` is True.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if return_folder:
        return check_slash(folder)


def h5_tree(val: h5py.File, pre=""):
    """
    Recursively prints the hierarchical structure of an h5py.File object.

    Args:
        val (h5py.File): The h5py.File object to print the structure of.
        pre (str, optional): The prefix string to add before each line. Defaults to "".
    """
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                h5_tree(val, pre + "    ")
            else:
                print(pre + "└── " + key + " (%d)" % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                h5_tree(val, pre + "│   ")
            else:
                print(pre + "├── " + key + " (%d)" % len(val))
