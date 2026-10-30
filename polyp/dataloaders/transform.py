import numpy as np


def collate_fn_w_transform(batch):
    image, label, name, domain = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    domain = np.stack(domain, 0)
    data_dict = {'data': image, 'mask': label, 'name': name, 'domain': domain}
    return data_dict


def collate_fn_wo_transform(batch):
    image, label, name, domain = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    domain = np.stack(domain, 0)
    data_dict = {'data': image, 'mask': label, 'name': name, 'domain': domain}
    return data_dict


def to_one_hot_list(mask_list):
    list = []
    for i in range(mask_list.shape[0]):
        mask = to_one_hot(mask_list[i].squeeze(0))
        list.append(mask)
    return np.stack(list, 0)


def to_one_hot(pre_mask, classes=2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [1, 0]
    mask[pre_mask == 2] = [1, 1]
    return mask.transpose(2, 0, 1)