import os
import pandas as pd


# def convert_labeled_list(root, csv_list):
#     img_list = list()
#     label_list = list()
#     for csv_file in csv_list:
#         data = pd.read_csv(os.path.join(root, csv_file))
#         img_list += data['image'].tolist()
#         label_list += data['mask'].tolist()
#     return img_list, label_list


def convert_labeled_list(root, csv_list):
    img_list = []
    label_list = []
    domain_list = []
    domain_map = {
        'RUNMC': 0,
        'BMC': 1,
        'I2CVB': 2,
        'UCL': 3,
        'BIDMC': 4,
        'HK': 5
    }
    for csv_file in csv_list:
        data = pd.read_csv(os.path.join(root, csv_file))
        # 依次判断关键字
        domain_label = None
        for key, value in domain_map.items():
            if key.lower() in csv_file.lower():
                domain_label = value
                break
        if domain_label is None:
            domain_label = 'Unknown'
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()
        domain_list += [domain_label] * len(data)
    return img_list, label_list, domain_list
