# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 14:00:15 2017

@author: shekh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import h2o

bogus_dealers_dir = r"E:\data\PreliminaryAnalysis\BogusDealers"

def load_returns():
    returns = pd.read_stata(bogus_dealers_dir+"\FeatureReturns.dta", convert_categoricals=False)
    returns['TIN_hash_byte']=returns['DealerTIN'].astype(str).apply(lambda x: ord(md5.new(x).digest()[0]))
    returns['RefundClaimedBoolean']=returns['RefundClaimed']>0
    return returns

def load_profile():
    profiles = pd.read_stata(bogus_dealers_dir+"\FeatureDealerProfiles.dta", convert_dates=False)
    return profiles

def load_registeredsales():
    transactions = pd.read_stata(bogus_dealers_dir+"\FeatureUnRegisteredSaleTransactions.dta", convert_dates=False)
    return transactions

def load_matchsales():
    SalesDiscrepancy=pd.read_stata(bogus_dealers_dir+"\SaleDiscrepancyAll.dta", convert_dates=False)
    return SalesDiscrepancy

def load_matchpurchases():
    PurchaseDiscrepancy=pd.read_stata(bogus_dealers_dir+"\PurchaseDiscrepancyAll.dta", convert_dates=False)
    return PurchaseDiscrepancy


def load_networkfeaturessales(TaxQuarter=9):
    NetworkFeaturesSales=pd.read_csv(bogus_dealers_dir+'\NetworkFeaturesSales'+str(TaxQuarter)+'.csv')
#    NetworkFeaturesSales=NetworkFeaturesSales.dropna(subset=['bogus_online'])
    NetworkFeaturesSales=NetworkFeaturesSales[['__id', 'TaxQuarter','pagerank','triangle_count',\
                                               'in_degree','out_degree']]
    NetworkFeaturesSales= NetworkFeaturesSales.rename(index=str, columns={"__id":"DealerTIN","component_id":"Sales_component_id",\
                                "pagerank":"Sales_pagerank","triangle_count":"Sales_triangle_count","in_degree":"Sales_in_degree",\
                                "out_degree":"Sales_out_degree","core_id":"Sales_core_id"})
    return NetworkFeaturesSales

def load_networkfeaturespurchases(TaxQuarter=9):
    NetworkFeaturesPurchases=pd.read_csv(bogus_dealers_dir+'\NetworkFeaturesPurchases'+str(TaxQuarter)+'.csv')
#    NetworkFeaturesPurchases=NetworkFeaturesPurchases.dropna(subset=['bogus_online'])
    NetworkFeaturesPurchases=NetworkFeaturesPurchases[['__id', 'TaxQuarter','pagerank','triangle_count','in_degree','out_degree']]
    NetworkFeaturesPurchases= NetworkFeaturesPurchases.rename(index=str, columns={"__id":"DealerTIN","component_id": "Purchases_component_id","pagerank":"Purchases_pagerank","triangle_count":"Purchases_triangle_count","in_degree":"Purchases_in_degree","out_degree":"Purchases_out_degree","core_id":"Purchases_core_id"})
    return NetworkFeaturesPurchases

def load_salenetwork():
    SaleNetworkQuarter=pd.DataFrame()
    for quarter in xrange(9,21):
        print "Quarter is"+str(quarter)
        SaleNetworkQuarterX=load_networkfeaturessales(quarter)
        SaleNetworkQuarter=SaleNetworkQuarter.append(SaleNetworkQuarterX,ignore_index=True)
    return SaleNetworkQuarter

def load_purchasenetwork():
    PurchaseNetworkQuarter=pd.DataFrame()
    for quarter in xrange(9,21):
        print "Quarter is"+str(quarter) 
        PurchaseNetworkQuarterX=load_networkfeaturespurchases(quarter)
        PurchaseNetworkQuarter=PurchaseNetworkQuarter.append(PurchaseNetworkQuarterX,ignore_index=True)
    return PurchaseNetworkQuarter

def load_h2odataframe_returns(returns):
    fr=h2o.H2OFrame(python_obj=returns)
    fr=set_return_factors(fr)
    fr=set_profile_factors(fr)
    fr=set_match_factors(fr)
    fr=set_transaction_factors(fr)
    fr=set_purchasenetwork_factors(fr)
    fr=set_salenetwork_factors(fr)
    return fr
    
def set_return_factors(fr):    
    fr['bogus_online'] = fr['bogus_online'].asfactor()
    fr['ZeroTaxCredit'] = fr['ZeroTaxCredit'].asfactor()
    fr['bogus_any'] = fr['bogus_any'].asfactor()
    fr['bogus_cancellation'] = fr['bogus_cancellation'].asfactor()
    fr['MoneyGroup'] = fr['MoneyGroup'].asfactor()
    fr['RefundClaimed'] = fr['RefundClaimed'].asfactor()
    fr['ZeroTurnover'] = fr['ZeroTurnover'].asfactor()
    fr['PositiveContribution'] = fr['PositiveContribution'].asfactor()
    fr['AllCentral'] = fr['AllCentral'].asfactor()
    fr['AllLocal'] = fr['AllLocal'].asfactor()
    fr['ZeroTax'] = fr['ZeroTax'].asfactor()
    fr['TaxQuarter'] = fr['TaxQuarter'].asfactor()
    fr['RefundClaimedBoolean'] = fr['RefundClaimedBoolean'].asfactor()
    return fr
    
def set_profile_factors(fr):
    fr['Nature']=fr['Nature'].asfactor()
    fr['Constitution']=fr['Constitution'].asfactor()
    fr['BooleanRegisteredCE']=fr['BooleanRegisteredCE'].asfactor()
    fr['BooleanServiceTax']=fr['BooleanServiceTax'].asfactor()
    fr['DummyManufacturer']=fr['DummyManufacturer'].asfactor()        
    fr['DummyWholeSaler']=fr['DummyWholeSaler'].asfactor()        
    fr['DummyInterStateSeller']=fr['DummyInterStateSeller'].asfactor()
    fr['DummyInterStatePurchaser']=fr['DummyInterStatePurchaser'].asfactor()    
    fr['DummyWorkContractor']=fr['DummyWorkContractor'].asfactor()    
    fr['DummyExporter']=fr['DummyExporter'].asfactor()
    fr['DummyOther']=fr['DummyOther'].asfactor()
    fr['DummyHotel']=fr['DummyHotel'].asfactor()
    fr['DummyECommerce']=fr['DummyECommerce'].asfactor()
    fr['DummyTelecom']=fr['DummyTelecom'].asfactor()
    fr['profile_merge']=fr['profile_merge'].asfactor()
    fr['StartYear']=fr['StartYear'].asfactor()
    fr['Ward']=fr['Ward'].asfactor()
    return fr    
    
def set_match_factors(fr):
    fr['purchasematch_merge']=fr['purchasematch_merge'].asfactor()
    fr['transaction_merge']=fr['transaction_merge'].asfactor()
    fr['salesmatch_merge']=fr['salesmatch_merge'].asfactor()
    return fr
    
def set_transaction_factors(fr):
    fr['_merge_purchasediscrepancy']=fr['_merge_purchasediscrepancy'].asfactor()
    fr['_merge_salediscrepancy']=fr['_merge_salediscrepancy'].asfactor()
    return fr
 
def set_purchasenetwork_factors(fr):
#    fr['Purchases_component_id']=fr['Purchases_component_id'].asfactor()
#    fr['Purchases_core_id']=fr['Purchases_core_id'].asfactor()
    fr['purchasenetwork_merge']=fr['purchasenetwork_merge'].asfactor()
    return fr


def set_salenetwork_factors(fr):
#    fr['Sales_component_id']=fr['Sales_component_id'].asfactor()
#    fr['Sales_core_id']=fr['Sales_core_id'].asfactor()
    fr['salesnetwork_merge']=fr['salesnetwork_merge'].asfactor()
    return fr
      
def divide_train_test(fr):
    """ train and test sets - outdated. In future - divide by DealerTIN """
    b = fr['TIN_hash_byte']
    train = fr[ b < 200 ]
    valid = fr[ (200 <= b) & (b < 232) ]
    test  = fr[ 232 <= b ]
    return train, valid, test

#def set_predictions(model,data):
#    X=model.predict(data)
#    X=X.as_data_frame(use_pandas=True)
#    Y=data.as_data_frame(use_pandas=True)
#    X=X.drop(['p0','predict'],axis=1)
#    result=pd.concat([Y,X],axis=1)
#    return result

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


#%%
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
    
    PredictionData=pd.merge(Z,Y, how='left', on=['DealerTIN','TaxQuarter'], indicator='prediction_merge')
    PredictionData.to_csv(path_or_buf=FilePath)