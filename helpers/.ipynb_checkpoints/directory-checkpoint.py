import os

def is_valid_entry(path):
    """Return True if path is a directory, .py, or .ipynb file."""
    if os.path.isdir(path):
        return True
    if path.endswith('.py') or path.endswith('.ipynb'):
        return True
    return False

def generate_ascii_tree(dir_path, prefix=''):
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        return

    # Filter out hidden files and __pycache__ folders
    entries = [e for e in entries if not e.startswith('.') and e != '__pycache__']
    entries = [e for e in entries if is_valid_entry(os.path.join(dir_path, e))]

    pointers = ['├──'] * (len(entries) - 1) + ['└──']

    for pointer, entry in zip(pointers, entries):
        path = os.path.join(dir_path, entry)
        print(prefix + pointer + ' ' + entry)
        if os.path.isdir(path):
            extension = '│   ' if pointer == '├──' else '    '
            generate_ascii_tree(path, prefix + extension)



