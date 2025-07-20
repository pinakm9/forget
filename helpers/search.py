import os
import json
import re

def highlight(text, search_str):
    """
    Highlight all occurrences of a given string in a given text.

    :param text: The text to highlight in.
    :param search_str: The string to highlight.
    :return: The text with all occurrences of `search_str` highlighted.
    """
    return re.sub(f'({re.escape(search_str)})', '\033[1m\\1\033[0m', text, flags=re.IGNORECASE)

def search_string_in_files(folder_path, search_str, context=1, max_matches=None, file_extensions=None, log_file=None):
    """
    Search for a string in a folder's files and print the matches with context.

    :param folder_path: The path to the folder to search in.
    :param search_str: The string to search for.
    :param context: The number of surrounding lines to include in the match.
    :param max_matches: The maximum number of matches to show. If None, all matches are shown.
    :param file_extensions: The file extensions to search in. If None, all files are searched.
    :param log_file: If not None, the results are saved to this file instead of printed to the console.
    """
    match_count = 0
    results = []

    if file_extensions:
        file_extensions = tuple(file_extensions)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.') or file == '__pycache__':
                continue
            if file_extensions and not file.endswith(file_extensions):
                continue

            file_path = os.path.join(root, file)
            try:
                if file.endswith('.ipynb'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        nb = json.load(f)
                    lines = []
                    for cell in nb.get("cells", []):
                        if cell.get("cell_type") == "code":
                            lines.extend(cell.get("source", []))
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
            except (UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError):
                continue

            for i, line in enumerate(lines):
                if search_str in line:
                    match_count += 1
                    if max_matches and match_count > max_matches:
                        break

                    snippet = []
                    start = max(0, i - context)
                    end = min(len(lines), i + context + 1)
                    snippet.append(f"\n📌 Match #{match_count} in {file_path} at line {i+1}:")
                    for j in range(start, end):
                        prefix = '➡' if j == i else '  '
                        content = highlight(lines[j].rstrip(), search_str) if j == i else lines[j].rstrip()
                        snippet.append(f"{prefix} {j+1:>4}: {content}")
                    results.append('\n'.join(snippet))

            if max_matches and match_count >= max_matches:
                break

    output = '\n'.join(results)

    if log_file:
        with open(log_file, 'w') as f:
            f.write(output)
        print(f"\n✅ Results saved to {log_file}")
    else:
        print(output)
        if max_matches and match_count > max_matches:
            print(f"\n⚠️ Showing only the first {max_matches} matches.")

            
            


def list_files_with_name_match(folder_path, search_str, extensions=None, recursive=True, case_insensitive=True):
    matches = []
    search_str = search_str.lower() if case_insensitive else search_str

    walk = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]

    for root, _, files in walk:
        for file in files:
            if extensions and not file.endswith(tuple(extensions)):
                continue

            name_to_check = file.lower() if case_insensitive else file
            if search_str in name_to_check:
                full_path = os.path.join(root, file)
                matches.append(full_path)

    # Print results
    for i, path in enumerate(matches, start=1):
        print(f"{i:2d}. {path}")

    print(f"\n✅ Found {len(matches)} matching files.")
    return matches



