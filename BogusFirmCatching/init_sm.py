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
import socket
from hashlib import md5

bogus_dealers_dir = r"E:\data\PreliminaryAnalysis\BogusDealers"
if socket.gethostname() == 'WIN-0HSQUR8Q0D3':
    bogus_dealers_dir = r"D:\data\PreliminaryAnalysis\BogusDealers"

def load_returns():
    returns = pd.read_stata(bogus_dealers_dir+"\FeatureReturns.dta", convert_categoricals=False)
    returns['TIN_hash_byte']=returns['DealerTIN'].astype(str).apply(lambda x: ord(md5(x).digest()[0]))
    returns['RefundClaimedBoolean']=returns['RefundClaimed']>0
    return returns

def load_profile():
    profiles = pd.read_stata(bogus_dealers_dir+"\FeatureDealerProfiles.dta", convert_dates=False)
    return profiles

#def load_registeredsales():
#    transactions = pd.read_stata(bogus_dealers_dir+"\FeatureUnRegisteredSaleTransactions.dta", convert_dates=False)
#    return transactions

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
        print "Quarter is "+str(quarter)
        SaleNetworkQuarterX=load_networkfeaturessales(quarter)
        SaleNetworkQuarter=SaleNetworkQuarter.append(SaleNetworkQuarterX,ignore_index=True)
    return SaleNetworkQuarter

def load_purchasenetwork():
    PurchaseNetworkQuarter=pd.DataFrame()
    for quarter in xrange(9,21):
        print "Quarter is "+str(quarter)
        PurchaseNetworkQuarterX=load_networkfeaturespurchases(quarter)
        PurchaseNetworkQuarter=PurchaseNetworkQuarter.append(PurchaseNetworkQuarterX,ignore_index=True)
    return PurchaseNetworkQuarter

def load_purchasedownstream():
    PurchaseDS=pd.read_stata(bogus_dealers_dir+"\FeatureDownStreamnessPurchases.dta", convert_dates=False)
    return PurchaseDS

def load_salesdownstream():
    SalesDS=pd.read_stata(bogus_dealers_dir+"\FeatureDownStreamnessSales.dta", convert_dates=False)
    return SalesDS

def set_downstream_factors(fr):
    fr['Missing_SalesDSUnTaxProp']=fr['Missing_SalesDSUnTaxProp'].asfactor()
    fr['Missing_SalesDSCreditRatio']=fr['Missing_SalesDSCreditRatio'].asfactor()
    fr['Missing_SalesDSVatRatio']=fr['Missing_SalesDSVatRatio'].asfactor()
    fr['Missing_MaxSalesProp']=fr['Missing_MaxSalesProp'].asfactor()
    fr['Missing_MaxPurchaseProp']=fr['Missing_MaxPurchaseProp'].asfactor()
    fr['Missing_PurchaseDSUnTaxProp']=fr['Missing_PurchaseDSUnTaxProp'].asfactor()
    fr['Missing_PurchaseDSCreditRatio']=fr['Missing_PurchaseDSCreditRatio'].asfactor()
    fr['Missing_PurchaseDSVatRatio']=fr['Missing_PurchaseDSVatRatio'].asfactor()
    fr['salesds_merge']=fr['salesds_merge'].asfactor()
    fr['purchaseds_merge']=fr['purchaseds_merge'].asfactor()
    return fr

def load_h2odataframe_returns(returns):
    """
    @returns is what's returned from load_everything()
    """
    fr=h2o.H2OFrame(python_obj=returns)
    print 'setting factors...'
    fr=set_return_factors(fr)
    fr=set_profile_factors(fr)
    fr=set_match_factors(fr)
    fr=set_transaction_factors(fr)
    fr=set_purchasenetwork_factors(fr)
    fr=set_salenetwork_factors(fr)
    fr=set_downstream_factors(fr)
    return fr

#def load_h2odataframe_returns_fromfile():
#    fr=h2o.upload_file(path=bogus_dealers_dir+'\FinalEverything_minusq12.csv')
#    fr=set_profile_factors(fr)
#    fr=set_match_factors(fr)
#    fr=set_transaction_factors(fr)
#    fr=set_purchasenetwork_factors(fr)
 #   fr=set_salenetwork_factors(fr)
#    fr=set_downstream_factors(fr)
#    return fr

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

