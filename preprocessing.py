import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Citation: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    import re
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"``", " ", string)
    string = re.sub(r"`", " " , string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()

def pad_sentences(sentence, max_length=0, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    Adapted from: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    if len(sentence) < max_length:
        difference = max_length - len(sentence)
        return sentence + [padding_word] * difference
    else:
        return sentence

def float_to_class(sentiment):
    """
    The sentiments are a floating point number [0.0, 1.0]. Here I convert them to sentiment classes [1,2,3,4,5] where
    1 = negative
    2 = somewhat negative
    3 = neutral
    4 = somewhat positive
    5 = positive
    """
    if (sentiment <= 1.0) & (sentiment > 0.8):
        return 4
    elif (sentiment <= 0.8) & (sentiment > 0.6):
        return 3
    elif (sentiment <= 0.6) & (sentiment > 0.4):
        return 2
    elif (sentiment <= 0.4) & (sentiment > 0.2):
        return 1
    else:
        return 0