import csv
import ast
import numpy as np

def load_perceptual_data():
    data = []
    with open('TrainSet.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                line[0:6] = [x.strip() for x in line[0:6]]
                line[2] = True if line[2]=='replicate' else False
                line[6:] = ['NaN' if x=='NaN' else int(x) for x in line[6:]]
                data.append(line)
            else:
                headers = line
    return headers,data

def load_molecular_data():
    data = []
    with open('molecular_descriptors_data.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                line[1:] = ['NaN' if x=='NaN' else float(x) for x in line[6:]]
                data.append(line)
            else:
                headers = line
    return headers,data

def leaderboard_CIDs():
    """Return CIDs for molecules that will be used for the leaderboard
    to determine the provisional leaders of the competition."""
    with open('CID_leaderboard.txt') as f:
        reader = csv.reader(f)
        result = [int(line[0]) for line in reader]
    return result

def testset_CIDs():
    """Return CIDs for molecules that will be used for final testing
    to determine the winners of the competition."""
    with open('CID_testset.txt') as f:
        reader = csv.reader(f)
        result = [int(line[0]) for line in reader]
    return result