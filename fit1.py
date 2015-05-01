import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

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
