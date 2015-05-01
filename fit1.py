import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.cross_validation import ShuffleSplit,cross_val_score

import scoring

def rfc_(X_train,Y_train,X_test_int,X_test_other,Y_test,max_features=1500,n_estimators=1000,max_depth=None,min_samples_leaf=1):
    print(max_features)
    def rfc_maker():
        return RandomForestRegressor(max_features=max_features,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     n_jobs=-1,
                                     oob_score=True,
                                     random_state=0)
    n_subjects = 49
    predicted_train = []
    observed_train = []
    predicted_test = []
    observed_test = []
    rfcs = {subject:rfc_maker() for subject in range(1,n_subjects+1)}
    for subject in range(1,n_subjects+1):
        print(subject)
        observed = Y_train[subject]
        rfc = rfcs[subject]
        rfc.fit(X_train,observed)
        #predicted = rfc.predict(X_train)
        predicted = rfc.oob_prediction_
        observed_train.append(observed)
        predicted_train.append(predicted)

        observed = Y_test[subject]
        rfc = rfcs[subject]
        if Y_train is Y_test: # OOB prediction  
            predicted = rfc.oob_prediction_
        else:
            predicted = rfc.predict(X_test_other)
            predicted_int = rfc.predict(X_test_int)
            predicted[:,0] = predicted_int[:,0]
        observed_test.append(observed)
        predicted_test.append(predicted)
    scores = {}
    for phase,predicted_,observed_ in [('train',predicted_train,observed_train),('test',predicted_test,observed_test)]:
        predicted = np.dstack(predicted_)
        observed = np.ma.dstack(observed_)
        predicted_mean = np.mean(predicted,axis=2,keepdims=True)
        regularize = 0.7
        predicted = regularize*(predicted_mean) + (1-regularize)*predicted
        score = scoring.score(predicted,observed,n_subjects=n_subjects)
        r_int = scoring.r('int',predicted,observed)
        r_ple = scoring.r('ple',predicted,observed)
        r_dec = scoring.r('dec',predicted,observed)
        print("For subchallenge 1, %s phase, score = %.2f (%.2f,%.2f,%.2f)" % (phase,score,r_int,r_ple,r_dec))
        scores[phase] = score
    return rfcs,scores['train'],scores['test']

# Show that random forest regrssion also works really well out of sample.  
from sklearn.cross_validation import ShuffleSplit,cross_val_score
def rfc_cv(X,Y,n_splits=5,max_features=1000,n_estimators=15,min_samples_leaf=1,regularize=[0.7,0.35,0.7]):
    test_size = 0.2
    n_molecules = X.shape[0]
    shuffle_split = ShuffleSplit(n_molecules,n_splits,test_size=test_size)
    test_size *= n_molecules
    rfcs = {}
    n_subjects = 49
    for subject in range(1,n_subjects+1):
        rfc = RandomForestRegressor(max_features=max_features,
                                    n_estimators=n_estimators,
                                    min_samples_leaf=min_samples_leaf,
                                    max_depth=None,
                                    oob_score=False,
                                    n_jobs=-1,
                                    random_state=0)
        rfcs[subject] = rfc
    rs = {'int':[],
          'ple':[],
          'dec':[]}
    scores = []
    for train_index,test_index in shuffle_split:
        predicted_list = []
        observed_list = []
        for subject in range(1,n_subjects+1):
            rfc = rfcs[subject]
            X_train = X[train_index]
            Y_train = Y[subject][train_index]
            rfc.fit(X_train,Y_train)
            X_test = X[test_index]
            predicted = rfc.predict(X_test)
            observed = Y[subject][test_index]
            predicted_list.append(predicted)
            observed_list.append(observed)
        observed = np.ma.dstack(observed_list)
        predicted = np.dstack(predicted_list)
        predicted_mean = predicted.mean(axis=2,keepdims=True)
        predicted_int = regularize[0]*(predicted_mean) + (1-regularize[0])*predicted
        predicted_ple = regularize[1]*(predicted_mean) + (1-regularize[1])*predicted
        predicted = regularize[2]*(predicted_mean) + (1-regularize[2])*predicted
        predicted[:,0,:] = predicted_int[:,0,:]
        predicted[:,1,:] = predicted_ple[:,1,:]
        score = scoring.score(predicted,observed)
        scores.append(score)
        for kind in ['int','ple','dec']:
            rs[kind].append(scoring.r(kind,predicted,observed))
    for kind in ['int','ple','dec']:
        rs[kind] = {'mean':np.mean(rs[kind]),'sem':np.std(rs[kind])/np.sqrt(n_splits)}
    scores = {'mean':np.mean(scores),'sem':np.std(scores)/np.sqrt(n_splits)}
    print("For subchallenge 1, using cross-validation with at least %d samples_per_leaf:" % min_samples_leaf)
    print("\tscore = %.2f+/- %.2f" % (scores['mean'],scores['sem']))
    for kind in ['int','ple','dec']:
        print("\t%s = %.2f+/- %.2f" % (kind,rs[kind]['mean'],rs[kind]['sem']))
            
    return scores,rs

