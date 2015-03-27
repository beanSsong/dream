import csv
import ast
import numpy as np
import types
from scipy.stats import pearsonr

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
                line[1:] = ['NaN' if x=='NaN' else float(x) for x in line[1:]]
                data.append(line)
            else:
                headers = line
    return headers,data

def test_CIDs(kind):
    assert kind in ['leaderboard','testset']
    """Return CIDs for molecules that will be used for:
        'leaderboard': the leaderboard to determine the provisional 
                       leaders of the competition.
        'testset': final testing to determine the winners 
                   of the competition."""
    with open('dilution_%s.txt' % kind) as f:
        reader = csv.reader(f,delimiter='\t')
        next(reader)
        data = [[int(line[0]),line[1]] for line in reader]
        for i,(CID,dilution) in enumerate(data):
            mag = max(-3,dilution2magnitude(dilution))
            data[i] = '%d_%g' % (CID,mag)
    return sorted(data)

def get_perceptual_matrices(perceptual_data,target_dilution=-3):
    perceptual_matrices = {}
    CIDs = []
    for row in perceptual_data:
        CID = int(row[0])
        CIDs.append(CID)
        dilution = dilution2magnitude(row[4])
        if target_dilution is None:
            key = '%d_%g' % (CID,dilution)#1.0/int(dilution.split('/')[1].replace(',','')))
        elif dilution == target_dilution:
            key = '%d_%g' % (CID,dilution)
        else:
            continue
        if key not in perceptual_matrices:
            perceptual_matrices[key] = np.ones((49,21))*np.NaN
        data = np.array([np.nan if _=='NaN' else int(_) for _ in row[6:]])
        subject = int(row[5])
        perceptual_matrices[key][subject-1,:] = data
                
    return perceptual_matrices

def get_molecular_vectors(molecular_data,CIDs):
    molecular_vectors = {}
    for row in molecular_data:
        CID = int(row[0])
        if CID in CIDs:
            molecular_vectors[CID] = np.array([np.nan if _=='NaN' else float(_) for _ in row[1:]])
    return molecular_vectors

def get_perceptual_vectors(perceptual_matrices,imputer=None,statistic='mean'):
    perceptual_vectors = {}
    for CID,matrix in perceptual_matrices.items():
        if imputer == 'zero':
            matrix[np.where(np.isnan(matrix))] = 0
        elif imputer:
            matrix = imputer.fit_transform(matrix) # Impute the NaNs.  
        if statistic == 'mean':
            perceptual_vectors[CID] = matrix.mean(axis=0)
        elif statistic == 'std':
            perceptual_vectors[CID] = matrix.std(axis=0)
        elif statistic is None:
            perceptual_vectors[CID] = {}
            for subject in range(1,matrix.shape[0]+1):        
                perceptual_vectors[CID][subject] = matrix[subject-1]
        else:
            raise Exception("Statistic '%s' not recognized" % statistic)
    return perceptual_vectors

def purge(this,from_that):
    from_that = {CID:value for CID,value in from_that.items() if CID not in this}
    return from_that

def retain(this,in_that):
    in_that = {CID:value for CID,value in in_that.items() if CID in this}
    return in_that

def dilution2magnitude(dilution):
    denom = dilution.replace('"','').replace("'","").split('/')[1].replace(',','')
    return np.log10(1.0/float(denom))
# Scoring for sub-challenge 1.

# Scoring.  

def r(kind,predicted,observed,n_subjects=49):
    # Predicted and observed should each be an array of 
    # molecules by perceptual descriptors by subjects
    
    r = 0.0
    for subject in range(n_subjects):
        p = predicted[:,:,subject]
        o = observed[:,:,subject]
        r += r2(kind,None,p,o)
    r /= n_subjects
    return r

def score(predicted,actual,n_subjects=49):
    """Final score for sub-challenge 1."""
    score = z('int',predicted,actual,n_subjects=n_subjects) \
          + z('ple',predicted,actual,n_subjects=n_subjects) \
          + z('dec',predicted,actual,n_subjects=n_subjects)
    return score/3.0

def z(kind,predicted,actual,n_subjects=49): 
    sigmas = {'int': 0.0187,
              'ple': 0.0176,
              'dec': 0.0042}

    sigma = sigmas[kind]
    shuffled_r = 0#r2(kind,predicted,shuffled)
    actual_r = r(kind,predicted,actual,n_subjects=n_subjects)
    return (actual_r - shuffled_r)/sigma

# Scoring for sub-challenge 2.  

def r2(kind,moment,predicted,observed):
    # Predicted and observed should each be an array of 
    # molecules by 2*perceptual descriptors (means then stds)
    if moment == 'mean':
        p = predicted[:,:21]
        o = observed[:,:21]
    elif moment == 'sigma':
        p = predicted[:,21:]
        o = observed[:,21:]
    elif moment is None:
        p = predicted
        o = observed
    else:
        raise ValueError('No such moment: %s' % moment)
    
    if kind=='int':
        p = p[:,0]
        o = o[:,0]
    elif kind=='ple':
        p = p[:,1]
        o = o[:,1]
    elif kind == 'dec':
        p = p[:,2:]
        o = o[:,2:]
    elif kind in range(19):
        p = p[:,2+kind]
        o = o[:,2+kind]
    else:
        raise ValueError('No such kind: %s' % kind)
    
    if len(p.shape)==1:
        r = pearsonr(p,o)[0]
    else:
        r = 0.0
        cols = p.shape[1]
        denom = 0.0
        for i in range(cols):
            p_ = p[:,i]
            o_ = o[:,i]
            r_ = pearsonr(p_,o_)[0]
            if np.isnan(r_):
                if np.std(p_)*np.std(o_) != 0:
                    print('WTF')
            else:
                r += r_
                denom += 1
        r /= denom
    return r

def score2(predicted,actual):
    """Final score for sub-challenge 2."""
    score = z2('int','mean',predicted,actual) \
          + z2('ple','mean',predicted,actual) \
          + z2('dec','mean',predicted,actual) \
          + z2('int','sigma',predicted,actual) \
          + z2('ple','sigma',predicted,actual) \
          + z2('dec','sigma',predicted,actual)
    return score/6.0

def z2(kind,moment,predicted,actual): 
    sigmas = {'int_mean': 0.1193,
              'ple_mean': 0.1265,
              'dec_mean': 0.0265,
              'int_sigma': 0.1194,
              'ple_sigma': 0.1149,
              'dec_sigma': 0.0281}

    sigma = sigmas[kind+'_'+moment]
    shuffled_r = 0#r2(kind,predicted,shuffled)
    actual_r = r2(kind,moment,predicted,actual)
    return (actual_r - shuffled_r)/sigma

def scorer2(estimator,inputs,actual):
    predicted = estimator.predict(inputs)
    return score2(predicted,actual)

