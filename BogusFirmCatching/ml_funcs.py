import pandas as pd
import numpy as np
from numpy import *
import h2o



def set_predictions(model,data):
    X=model.predict(data)
    X=X.as_data_frame(use_pandas=True)
#    Y=data.as_data_frame(use_pandas=True)
    X=X.drop(['p0','predict'],axis=1)
#    result=pd.concat([Y,X],axis=1)
    return X

def set_prediction_name(data,original,new):
    data.rename(index=str,columns={original:new},inplace=True)
    return data


def generate_predictions(models,ValidationData,FilePath,ColumnTitle):
    PredictionDataModels = []
    for i in xrange(len(models)):
        PredictionDataModels.append(set_predictions(models[i],ValidationData))
        PredictionDataModels[i]=set_prediction_name(PredictionDataModels[i],'p1',ColumnTitle+'{}'.format(i+1))
    
    Y = pd.concat(PredictionDataModels,axis=1,ignore_index=False)
    #Y=Y.as_data_frame(use_pandas=True)
    
    Z=ValidationData.as_data_frame(use_pandas=True)
    Z=Z[['DealerTIN','TaxQuarter','bogus_online','bogus_cancellation','profile_merge','transaction_merge','salesmatch_merge','purchasematch_merge','purchasenetwork_merge','salesnetwork_merge']]
    Z.index=Z.index.map(unicode)
    
    PredictionData=pd.concat([Z,Y],axis=1,ignore_index=False)
    #PredictionData=pd.merge(Z,Y, how='left', on=['DealerTIN','TaxQuarter'], indicator='prediction_merge')
    PredictionData.to_csv(path_or_buf=FilePath)# -*- coding: utf-8 -*-

def divide_train_test(fr):
    """ train and test sets - outdated. In future - divide by DealerTIN """
    b = fr['TIN_hash_byte']
    train = fr[ b < 200 ]
    valid = fr[ (200 <= b) & (b < 232) ]
    test  = fr[ 232 <= b ]
    return train, valid, test
