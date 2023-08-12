import os

# This script defines functions to calculate missing items between a directory containing image files and a directory containing JSON files.
# The '_get_image_filenames' function extracts image file names from a given directory.
# The '_get_json_filenames' function extracts JSON file names and matches them with image file names.
# The '_find_list_difference_loop_with_order' function finds the difference between two lists while preserving the order of items.
# The 'calculate_missing_items' function uses the above functions to determine missing items in either the image or JSON directory.

def _get_image_filenames(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_filenames = []

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            name_without_extension = os.path.splitext(filename)[0]
            image_filenames.append(name_without_extension)

    return image_filenames


def _get_json_filenames(directory):
    json_extensions = ['.json']
    image_extensions = ['.jpg', '.jpeg', '.png']
    json_filenames = []

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in json_extensions):
            name_without_json_extension = os.path.splitext(filename)[0]
            if any(name_without_json_extension.lower().endswith(ext) for ext in image_extensions):
                name_without_image_extension=os.path.splitext(name_without_json_extension)[0]
                json_filenames.append(name_without_image_extension)

    return json_filenames


def _find_list_difference_loop_with_order(list1, list2):
    difference = []
    for item in list1:
        if item not in list2:
            difference.append(item)
    return difference


def calculate_missing_items(image_path, json_file_path):
    image_names = _get_image_filenames(image_path)
    filenames = _get_json_filenames(json_file_path)

    result1 = _find_list_difference_loop_with_order(filenames, image_names)
    result2 = _find_list_difference_loop_with_order(image_names, filenames)

    combined_items = result1 + result2
    unique_items = list(set(combined_items))  
    return unique_items

