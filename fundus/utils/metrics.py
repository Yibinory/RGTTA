import numpy as np

def data_process(pred, label, threshold=0.5):
    pred = np.array(pred)
    label = np.array(label)

    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    return pred.astype(np.uint8), label.astype(np.uint8)


def dice_metric(pred, label):
    batch_size = pred.shape[0]
    disc_dices, cup_dices = [], []
    smooth = 1e-6

    for batch in range(batch_size):
        disc_intersection = (pred[batch][0] * label[batch][0]).sum()
        disc_dice = (2 * disc_intersection + smooth) / (pred[batch][0].sum() + label[batch][0].sum() + smooth)
        cup_intersection = (pred[batch][-1] * label[batch][-1]).sum()
        cup_dice = (2 * cup_intersection + smooth) / (pred[batch][-1].sum() + label[batch][-1].sum() + smooth)

        disc_dices.append(disc_dice)
        cup_dices.append(cup_dice)

    return np.mean(disc_dices), np.mean(cup_dices)

def calculate_metrics(test, reference):
    test, reference = data_process(pred=test, label=reference, threshold=0.5)
    disc_dice, cup_dice = dice_metric(test, reference)
    return [disc_dice, cup_dice]