import csv
import ast
import time

import numpy as np

from __init__ import *

def load_perceptual_data(kind):
    if kind == 'training':
        kind = 'TrainSet'
    elif kind == 'leaderboard':
        kind = 'LeaderboardSet'
    else:
        raise ValueError("No such kind: %s" % kind)
    
    data = []
    with open('%s.txt' % kind) as f:
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

def format_leaderboard_perceptual_data():
    f_new = open('LeaderboardSet.txt','w')
    writer = csv.writer(f_new,delimiter="\t")
    headers,_ = load_perceptual_data('training')
    descriptors = headers[6:]
    writer.writerow(headers)
    CID_dilutions = get_CID_dilutions('leaderboard',target_dilution='raw')
    dilutions = {}
    for CID_dilution in CID_dilutions:
        CID,mag,high = CID_dilution.split('_')
        dilutions[int(CID)] = {'dilution':('1/%d' % 10**(-int(mag))),
                               'high':int(high)}
    lines_new = {}
    
    with open('LBs1.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                CID,subject,descriptor,value = line
                CID = int(CID)
                subject = int(subject)
                if descriptor == 'INTENSITY/STRENGTH':
                    dilution = '1/1000'
                    high = 1-dilutions[CID]['high']
                else:
                    high = dilutions[CID]['high']
                    dilution = dilutions[CID]['dilution'] if high else '1/1000'
                line_id = '%d_%d_%d' % (CID,subject,dilution2magnitude(dilution))
                if line_id not in lines_new:
                    lines_new[line_id] = [CID,'N/A',0,'high' if high else 'low',dilution,subject]+['NaN']*21
                lines_new[line_id][6+descriptors.index(descriptor.strip())] = value

    with open('leaderboard_set.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num,line in enumerate(reader):
            if line_num > 0:
                CID,subject,descriptor,value = line
                CID = int(CID)
                subject = int(subject)
                if descriptor == 'INTENSITY/STRENGTH':
                    dilution = dilutions[CID]['dilution']
                    high = dilutions[CID]['high']
                else:
                    high = 1-dilutions[CID]['high']
                    dilution = dilutions[CID]['dilution'] if high else '1/1000'
                line_id = '%d_%d_%d' % (CID,subject,dilution2magnitude(dilution))
                if line_id not in lines_new:
                    lines_new[line_id] = [CID,'N/A',0,'high' if high else 'low',dilution,subject]+['NaN']*21
                lines_new[line_id][6+descriptors.index(descriptor.strip())] = value


    for line_id in sorted(lines_new,key=lambda x:[int(_) for _ in x.split('_')]):
        line = lines_new[line_id]
        writer.writerow(line)
    f_new.close()

def load_leaderboard_perceptual_data(target_dilution=None):
    """Loads directly into Y"""
    if target_dilution == 'gold':
        # For actual testing, use 1/1000 dilution for intensity and
        # high dilution for everything else.  
        Y = load_leaderboard_perceptual_data(target_dilution='high')
        intensity = load_leaderboard_perceptual_data(target_dilution=-3)
        Y['mean_std'][:,0] = intensity['mean_std'][:,0]
        Y['mean_std'][:,21] = intensity['mean_std'][:,21]
        for i in range(len(Y['subject'])):
            Y['subject'][i][:,0] = intensity['subject'][i][:,0]
            Y['subject'][i][:,21] = intensity['subject'][i][:,21]
        return Y
    #assert target_dilution in [None,'low','high']
    perceptual_headers, _ = load_perceptual_data()
    descriptors = perceptual_headers[6:]
    CIDs = get_CID_dilutions('leaderboard',target_dilution=target_dilution)
    CIDs_all = get_CID_dilutions('leaderboard',target_dilution=None)
    CIDs = [int(_.split('_')[0]) for _ in CIDs] # Keep only CIDs, not dilutions.  
    n_molecules = len(CIDs)
    Y = {#'mean':np.zeros((n_molecules,21)),
         #'std':np.zeros((n_molecules,21)),
         'mean_std':np.zeros((n_molecules,42)),
         'subject':{i:np.zeros((n_molecules,21)) for i in range(1,50)}}
    if target_dilution not in ['low','high',None]:
        CID_ranks = get_CID_rank('leaderboard',dilution=target_dilution)
    if target_dilution != 'low':
        with open('LBs1.txt') as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num,line in enumerate(reader):
                if line_num > 0:
                    CID,subject,descriptor,value = line
                    if target_dilution not in ['low','high',None] and CID_ranks[int(CID)]!=1:
                        continue
                    Y['subject'][int(subject)][CIDs.index(int(CID)),descriptors.index(descriptor.strip())] = value
        with open('LBs2.txt') as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num,line in enumerate(reader):
                if line_num > 0:
                    CID,descriptor,mean,std = line
                    if target_dilution not in ['low','high',None] and CID_ranks[int(CID)]!=1:
                        continue
                    #Y['mean'][CIDs.index(CID),descriptors.index(descriptor.strip())] = mean
                    #Y['std'][CIDs.index(CID),descriptors.index(descriptor.strip())] = std
                    Y['mean_std'][CIDs.index(int(CID)),descriptors.index(descriptor.strip())] = mean
                    Y['mean_std'][CIDs.index(int(CID)),21+descriptors.index(descriptor.strip())] = std
                    
    if target_dilution != 'high':
        with open('leaderboard_set.txt') as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num,line in enumerate(reader):
                if line_num > 0:
                    CID,subject,descriptor,value = line
                    if target_dilution not in ['low','high',None] and CID_ranks[int(CID)]!=0:
                        continue
                    Y['subject'][int(subject)][CIDs.index(int(CID)),descriptors.index(descriptor.strip())] = value
            z = np.dstack([_ for _ in Y['subject'].values()])
            mask = np.zeros(z.shape)
            mask[np.where(np.isnan(z))] = 1
            z = np.ma.array(z,mask=mask)
            y_mean = z.mean(axis=2)
            y_std = z.std(axis=2,ddof=1)
        for CID in CID_ranks:
            if CID_ranks[CID] == 0:
                row = CIDs.index(CID)
                col = descriptors.index(descriptor.strip())
                Y['mean_std'][row,:21] = y_mean[row,:]
                Y['mean_std'][row,21:] = y_std[row,:]
    Y['mean_std'] = Y['mean_std'].round(2)
    return Y

def load_molecular_data():
    with open('molecular_descriptors_data.txt') as f:
        reader = csv.reader(f, delimiter="\t")
        data = []
        for line_num,line in enumerate(reader):
            if line_num > 0:
                line[1:] = ['NaN' if x=='NaN' else float(x) for x in line[1:]]
                data.append(line)
            else:
                headers = line
    return headers,data

def get_CID_dilutions(kind,target_dilution=None):
    assert kind in ['training','leaderboard','testset']
    """Return CIDs for molecules that will be used for:
        'leaderboard': the leaderboard to determine the provisional 
                       leaders of the competition.
        'testset': final testing to determine the winners 
                   of the competition."""
    if kind == 'training':
        data = []
        _,lines = load_perceptual_data(kind)
        for line in lines[1:]:
            CID = int(line[0])
            dilution = line[4]
            mag = dilution2magnitude(dilution)
            high = line[3] == 'high'
            if target_dilution == 'high' and not high:
                continue
            if target_dilution == 'low' and not low:
                continue
            elif target_dilution not in [None,'high','low'] and \
                 mag != target_dilution:
                 continue
            data.append("%d_%g_%d" % (CID,mag,high))
        data = list(set(data))
    else:
        with open('dilution_%s.txt' % kind) as f:
            reader = csv.reader(f,delimiter='\t')
            next(reader)
            lines = [[int(line[0]),line[1]] for line in reader]
            data = []
            for i,(CID,dilution) in enumerate(lines):
                mag = dilution2magnitude(dilution)
                high = (mag > -3)
                if target_dilution == 'high':
                    if high:
                        data.append('%d_%g_%d' % (CID,mag,1))
                    else:
                        data.append('%d_%g_%d' % (CID,-3,1))
                elif target_dilution == 'low':
                    if not high:
                        data.append('%d_%g_%d' % (CID,mag,0))
                    else:
                        data.append('%d_%g_%d' % (CID,-3,0))
                elif target_dilution is None:
                    data.append('%d_%g_%d' % (CID,mag,high))
                    data.append('%d_%g_%d' % (CID,-3,1-high))
                elif target_dilution in [mag,'raw']:
                    data.append('%d_%g_%d' % (CID,mag,high))
                elif target_dilution == -3:
                    data.append('%d_%g_%d' % (CID,-3,1-high))
 
    return sorted(data,key=lambda x:[int(_) for _ in x.split('_')])

def get_CIDs(kind):
    CID_dilutions = get_CID_dilutions(kind)
    CIDs = [int(_.split('_')[0]) for _ in CID_dilutions]
    return sorted(list(set(CIDs)))

def get_CID_rank(kind,dilution=-3):
    """Returns CID dictionary with 1 if -3 dilution is highest, 
    0 if it is lowest, -1 if it is not present.
    """

    CID_dilutions = get_CID_dilutions(kind)
    CIDs = set([int(_.split('_')[0]) for _ in CID_dilutions])
    result = {}
    for CID in CIDs:
        high = '%d_%g_%d' % (CID,dilution,1)
        low = '%d_%g_%d' % (CID,dilution,0)
        if high in CID_dilutions:
            result[CID] = 1
        elif low in CID_dilutions:
            result[CID] = 0
        else:
            result[CID] = -1
    return result

def dilution2magnitude(dilution):
    denom = dilution.replace('"','').replace("'","").split('/')[1].replace(',','')
    return np.log10(1.0/float(denom))

"""Output"""

# Write predictions for each subchallenge to a file.  
def open_prediction_file(subchallenge,kind,name):
    f = open('submissions/challenge_%d_%s_%s.txt' % (subchallenge,kind,name),'w')
    writer = csv.writer(f,delimiter='\t')
    return f,writer

def write_prediction_files(Y,kind,subchallenge,name):
    f,writer = open_prediction_file(subchallenge,kind,name=name)
    CIDs = get_CIDs(kind)
    perceptual_headers, _ = load_perceptual_data('training')

    # Subchallenge 1.
    if subchallenge == 1:
        writer.writerow(["#oID","individual","descriptor","value"])
        for subject in range(1,NUM_SUBJECTS+1):
            for j in range(NUM_DESCRIPTORS):
                for i,CID in enumerate(CIDs):
                    descriptor = perceptual_headers[-NUM_DESCRIPTORS:][j]
                    value = Y['subject'][subject][i,j].round(3)
                    writer.writerow([CID,subject,descriptor,value])
        f.close()
    
    # Subchallenge 2.
    elif subchallenge == 2:
        writer.writerow(["#oID","descriptor","value","sigma"])
        for j in range(NUM_DESCRIPTORS):
            for i,CID in enumerate(CIDs):
                descriptor = perceptual_headers[-NUM_DESCRIPTORS:][j]
                value = Y['mean_std'][i,j].round(3)
                sigma = Y['mean_std'][i,j+NUM_DESCRIPTORS].round(3)
                writer.writerow([CID,descriptor,value,sigma])
        f.close()

def make_prediction_files(rfcs,X_int,X_other,target,subchallenge,trans_weight=0.5,regularize=[0.7,0.35,0.6],name=None):
    if len(regularize)==1 and type(regularize)==list:
        regularize = regularize*3
    if name is None:
        name = '%d' % time.time()
    Y = {'subject':{'mean':0}}
    
    if subchallenge == 1:
        y_list = [rfcs[1][subject].predict(X_other) \
                  for subject in range(1,NUM_SUBJECTS+1)]
        Y['subject']['mean'] = np.mean(np.dstack(y_list),axis=2)
        for subject in range(1,NUM_SUBJECTS+1):
            # dec
            Y['subject'][subject] = (1-regularize[2])*rfcs[1][subject].predict(X_other) \
                                      + regularize[2]*Y['subject']['mean']
            # ple
            Y['subject'][subject][:,1] = (1-regularize[1])*rfcs[1][subject].predict(X_other)[:,1] \
                                      + regularize[1]*Y['subject']['mean'][:,1]
        del Y['subject']['mean']

        y_list = [rfcs[1][subject].predict(X_int) \
                  for subject in range(1,NUM_SUBJECTS+1)]
        Y['subject']['mean'] = np.mean(np.dstack(y_list),axis=2)
        for subject in range(1,NUM_SUBJECTS+1):
            # int
            Y['subject'][subject][:,0] = (1-regularize[0])*rfcs[1][subject].predict(X_int)[:,0] \
                                      + regularize[0]*Y['subject']['mean'][:,0]
        del Y['subject']['mean']

    if subchallenge == 2:
        kinds = ['int','ple','dec']
        moments = ['mean','sigma']
        ys = {kind:{} for kind in kinds}
        for kind in ['int','ple','dec']:
            X = X_int if kind=='int' else X_other
            for moment in ['mean','sigma']:
                ys[kind][moment] = rfcs[kind][moment].predict(X)
        y = ys['int']['mean'].copy()
        y[:,1] = ys['ple']['mean'][:,1]
        y[:,2:21] = ys['ple']['mean'][:,2:21]
        y[:,21] = ys['ple']['mean'][:,21]
        y[:,22] = ys['ple']['mean'][:,22]
        y[:,23:] = ys['ple']['mean'][:,23:]
        Y['mean_std'] = y
        
    write_prediction_files(Y,target,subchallenge,name)
    return Y

