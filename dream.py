import csv
import ast

def load_perceptual_data():
    data = []
    with open('TrainSet.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                line[0:6] = [x.strip() for x in line[0:6]]
                line[2] = True if line[2]=='replicate' else False
                line[6:] = [9999 if x=='NaN' else int(x) for x in line[6:]]
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
                line[1:] = [9999 if x=='NaN' else float(x) for x in line[6:]]
                data.append(line)
            else:
                headers = line
    return headers,data