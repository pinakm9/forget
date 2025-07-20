# script for replacing sphere with circle in file content
import os 
from pathlib import Path

# exts = ['.ipynb', '.txt', '.py']
# word = 'sphere'
# replace = 'circle'
# cur_path = os.path.realpath(__file__)
# repo = str(Path(os.path.dirname(cur_path)).parent)
# print(repo)
# file_counter = 0

# for root, dirs, files in os.walk(repo):
#     for file in files:
#         for ext in exts:
#             if file.endswith(ext):
#                 file = os.path.join(repo, root, file)
#                 if file != cur_path:
#                     with open(file, 'r') as f:
#                         data = f.read()
#                     if word in data:
#                         with open(file, 'w') as f:
#                             f.write(data.replace(word, replace))
#                         file_counter += 1
#                         print('Files updated ---> {}'.format(file_counter), end='\r')
#                         break



def replace_text_in_files(root_dir, target_word, replacement_word, file_extensions=None, case_insensitive=True):
    updated_files = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('.'):
                continue
            if file_extensions and not file.endswith(tuple(file_extensions)):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, FileNotFoundError):
                continue  # Skip unreadable files

            original_content = content

            if case_insensitive:
                pattern = target_word.lower()
                content_lower = content.lower()

                if pattern in content_lower:
                    # Manual replacement with original case preserved before and after
                    start = 0
                    result = ''
                    while True:
                        idx = content_lower.find(pattern, start)
                        if idx == -1:
                            result += content[start:]
                            break
                        result += content[start:idx] + replacement_word
                        start = idx + len(target_word)
                    content = result
            else:
                if target_word in content:
                    content = content.replace(target_word, replacement_word)

            # Write only if modified
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_files.append(file_path)

    # Report
    for i, path in enumerate(updated_files, 1):
        print(f"{i:2d}. Updated: {path}")

    print(f"\n✅ Done. Updated {len(updated_files)} files.")
    return updated_files


