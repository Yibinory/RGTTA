import os
import pandas as pd


def convert_labeled_list(root, csv_list):
    img_list = []
    label_list = []
    domain_list = []
    domain_map = {}
    if "Processed_Fundus_Images" in root:
        domain_map = {
            'RIM_ONE': 'A',
            'REFUGE_Valid': 'D',
            'REFUGE': 'B',
            'ORIGA': 'C',
            'Drishti_GS': 'E'
        }
    else:
        domain_map = {
            'BKAI': 'A',
            'CVC-ClinicDB': 'B',
            'ETIS-LaribPolypDB': 'C',
            'Kvasir-SEG': 'D'
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


