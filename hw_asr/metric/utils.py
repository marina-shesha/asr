# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    splited_target = target_text.split(' ')
    if len(splited_target) == 0:
        return 1
    return editdistance.distance(splited_target, predicted_text.split(' '))/len(splited_target)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, predicted_text)/len(target_text)