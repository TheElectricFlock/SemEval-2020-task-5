import csv

def f1(predictions, gold):
    """
    Calculates F1 score(a.k.a. DICE)
    
    Args:
        predictions: a list of predicted offsets
        gold: a list of offsets serving as the ground truth
        
    Returns: 
        a float score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions) == 0 else 0
    if len(predictions) == 0:
        return 0
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)
    
def read_text_data(filename):
    """
    Reads a csv file to extract text.
    
    Args:
        filename: a string specifying the filename / path
        
    Returns:
        A list of text sentences
    """
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            data.append(row['text'])
    csvfile.close()
    return data

def read_data_span(filename):
    """
    Reads a csv file to extract the toxic spans.
    
    Args:
        filename: a string specifying the filename / path
        
    Returns:
        data: a list of strings of the toxic chars, 
        will look like '[1,2,3]' so it'll have to be split
        
    """
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            data.append(row['span'])
    csvfile.close()
    return data