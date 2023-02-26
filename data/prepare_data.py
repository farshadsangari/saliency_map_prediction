import os
import re
import pandas as pd


def get_file_list(data_path):
    data_list = []
    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".jpg")):
                    data_list.append(os.path.join(subdir, file))
    data_list.sort()
    if not data_list:
        raise FileNotFoundError("No data was found")
    return data_list


def train_val_pathes(x_path, y_path, regex_for_category):
    x_path = get_file_list(x_path)
    x_path = [data.replace("\\", "/") for data in x_path]
    x_path = [path for path in x_path if re.findall("Output", path) == []]

    y_path = get_file_list(y_path)
    y_path = [data.replace("\\", "/") for data in y_path]
    y_path = [path for path in y_path if re.findall("Output", path) == []]

    path_data = pd.DataFrame(x_path, y_path).reset_index(level=0)
    path_data.columns = ["y_path", "x_path"]
    # regex_for_category = \/content\/drive\/MyDrive\/Database\/HW2_deep\/data_Q1\/trainSet\/Stimuli\/(.*)\/\d*\.jpg
    path_data["category"] = path_data["x_path"].apply(
        lambda x: re.search(regex_for_category, x).group(1)
    )

    frames = []
    classes = path_data.category.unique()
    sample_size = 15

    for i in classes:
        g = path_data[path_data.category == i].sample(sample_size)
        frames.append(g)

    equally_sampled = pd.concat(frames)

    train_path_index = [
        path
        for path in list(path_data.index)
        if path not in list(equally_sampled.index)
    ]
    train_path = path_data.loc[train_path_index, :].reset_index(drop=True)

    val_path_index = list(equally_sampled.index)
    val_path = path_data.loc[val_path_index, :].reset_index(drop=True)

    return [train_path, val_path]
