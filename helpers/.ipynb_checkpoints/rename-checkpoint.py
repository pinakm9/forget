# script for renaming the circle files into circle files

import os 
from pathlib import Path

# word = 'circle'
# replace = 'circle'
# repo = str(Path(os.path.dirname(os.path.realpath(__file__))).parent)
# print(repo)
# file_counter = 0

# for root, dirs, files in os.walk(repo):
#     for file in files:
#         if word in file:    
#             old_file = os.path.join(repo, root, file)
#             #print(old_file)
#             new_file = os.path.join(repo, root, file.replace(word, replace))
#             #print(new_file)
#             os.replace(old_file, new_file)
#             file_counter += 1
#             print('Files renamed ---> {}'.format(file_counter), end='\r')



def rename_files_and_folders(root_dir, target_word, replacement_word, case_insensitive=True):
    renamed = []

    if case_insensitive:
        target_word_lower = target_word.lower()

    # Walk from the deepest level first to avoid renaming parent folders before their contents
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # Rename files
        for name in files:
            original = name
            match = (target_word_lower in name.lower()) if case_insensitive else (target_word in name)
            if match:
                new_name = name.replace(target_word, replacement_word)
                if case_insensitive:
                    # Case-insensitive replacement: rebuild manually
                    name_lower = name.lower()
                    idx = name_lower.find(target_word_lower)
                    new_name = name[:idx] + replacement_word + name[idx+len(target_word):]

                os.rename(os.path.join(root, name), os.path.join(root, new_name))
                renamed.append((os.path.join(root, original), os.path.join(root, new_name)))

        # Rename directories
        for name in dirs:
            original = name
            match = (target_word_lower in name.lower()) if case_insensitive else (target_word in name)
            if match:
                new_name = name.replace(target_word, replacement_word)
                if case_insensitive:
                    name_lower = name.lower()
                    idx = name_lower.find(target_word_lower)
                    new_name = name[:idx] + replacement_word + name[idx+len(target_word):]

                os.rename(os.path.join(root, name), os.path.join(root, new_name))
                renamed.append((os.path.join(root, original), os.path.join(root, new_name)))

    # Report
    for i, (old, new) in enumerate(renamed, 1):
        print(f"{i:2d}. Renamed: {old} → {new}")

    print(f"\n✅ Done. Renamed {len(renamed)} items.")
    return renamed


