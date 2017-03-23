# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 13:50:50 2017
Creating the ML code
@author: Shekhar Mittal
"""
#%%
# Initiatlising the file
execfile(r'init_sm.py')
execfile(r'Graphs\graphsetup.py')
execfile(r'ml_funcs.py')

import statsmodels.api as sm
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from numpy import *
from bokeh.palettes import *
import md5

def init():
    global sr,fr,share_cols
    # <hack> regarding output to make h2o work in IDLE
    class PseudoTTY(object):
        def __init__(self, underlying):
            underlying.encoding = 'cp437'
            self.__underlying = underlying
        def __getattr__(self, name):
            return getattr(self.__underlying, name)
        def isatty(self):
            return True

    import sys
    sys.stdout = PseudoTTY(sys.stdout)
    # </hack>
    h2o.init(nthreads = -1,max_mem_size="48G")
    h2o.remove_all()

#defining a function to shut down the code
def shutdown_sm():
    h2o.cluster().shutdown()
    %clear
    %reset -f

init()
#%%
# Features that the Tax Authority uses
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross',\
                  'TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment',\
                  'TotalReturnCount']
# Features that we think can be important
important_features =  ['PositiveContribution','InterstateRatio','RefundClaimedBoolean',\
                       'MoneyGroup','CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
remaining_features = ['PercPurchaseUnregisteredDealer','PercValueAdded','AllCentral', \
                      'AllLocal', 'ZeroTax', 'ZeroTurnover', 'ZeroTaxCredit','TotalPurchases']

return_features=basic_features+important_features+remaining_features


# Dealer Features
dealer_features=['profile_merge','Ward','Constitution','BooleanRegisteredIEC',\
                 'BooleanRegisteredCE','BooleanServiceTax','StartYear',\
                 'DummyManufacturer','DummyRetailer','DummyWholeSaler',\
                 'DummyInterStateSeller','DummyInterStatePurchaser','DummyWorkContractor',\
                 'DummyImporter','DummyExporter','DummyOther','DummyHotel','DummyECommerce',\
                 'DummyTelecom']


# Transaction Features
transaction_features=['transaction_merge','UnTaxProp']
# Match Features
#match_features=['purchasematch_merge','salesmatch_merge', 'PurchasesAvgMatch', 'PurchasesAvgMatch3', 'PurchasesNameAvgMatch','SalesAvgMatch','SalesAvgMatch3','SalesNameAvgMatch' ]
match_features=['purchasematch_merge','salesmatch_merge','SaleMyCountDiscrepancy', \
                'SaleOtherCountDiscrepancy','SaleMyTaxDiscrepancy','SaleOtherTaxDiscrepancy',\
                'SaleDiscrepancy','absSaleDiscrepancy', '_merge_salediscrepancy', \
                'PurchaseMyCountDiscrepancy','PurchaseOtherCountDiscrepancy',\
                'PurchaseMyTaxDiscrepancy','PurchaseOtherTaxDiscrepancy',\
                'PurchaseDiscrepancy','absPurchaseDiscrepancy' ]

network_features=[ u'Purchases_pagerank', u'Purchases_triangle_count', u'Purchases_in_degree',\
                  u'Purchases_out_degree', u'purchasenetwork_merge', u'Sales_pagerank',\
                  u'Sales_triangle_count', u'Sales_in_degree', u'Sales_out_degree',\
                  u'salesnetwork_merge']

ds_features=[u'purchaseds_merge', u'MaxPurchaseProp', u'PurchaseDSUnTaxProp',\
             u'PurchaseDSCreditRatio', u'PurchaseDSVatRatio', u'Missing_PurchaseDSUnTaxProp', \
             u'Missing_PurchaseDSCreditRatio', u'Missing_PurchaseDSVatRatio', u'TotalSellers',\
             u'salesds_merge', u'MaxSalesProp', u'SalesDSUnTaxProp',\
             u'SalesDSCreditRatio', u'SalesDSVatRatio', u'Missing_SalesDSUnTaxProp', \
             u'Missing_SalesDSCreditRatio', u'Missing_SalesDSVatRatio', u'TotalBuyers']
       
all_network_features=transaction_features+match_features+network_features+ds_features
#%%
rf_models = [H2ORandomForestEstimator(
        model_id="rf_v{}".format(i),
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000) \
    for i in xrange(1,8)]
#%%
#Merging all data into one file
# load returns data
ReturnsAll=load_returns()
ReturnsAll['DealerTIN']=pd.to_numeric(ReturnsAll['DealerTIN'])
ReturnsAll['TaxQuarter']=pd.to_numeric(ReturnsAll['TaxQuarter'])

# load profiles data
Profiles=load_profile()
Profiles['DealerTIN']=pd.to_numeric(Profiles['DealerTIN'],errors='coerce')


#Merge returns data with profile data
ReturnsAllWithProfiles=pd.merge(ReturnsAll, Profiles, how='left', on=['DealerTIN'], indicator='profile_merge')

#save returns only from year 3 onwards (inclusive)
ReturnsPostY2WithProfiles=ReturnsAllWithProfiles[ReturnsAllWithProfiles['TaxQuarter']>8]

# load data related to calculating percentage sales made to registered firms
#Transactions=load_registeredsales()

# Merge Returns+Profiles+RegisteredSalesData
#ReturnsPostY2WithProfilesWithTransactions = pd.merge(ReturnsPostY2WithProfiles, Transactions, how='left', on=['DealerTIN','TaxQuarter'], indicator='transaction_merge')
#ReturnsPostY2WithProfilesWithTransactions['UnTaxProp']=0
#ReturnsPostY2WithProfilesWithTransactions['RTaxProp']=0

#ReturnsPostY2WithProfilesWithTransactions['UnTaxProp']=ReturnsPostY2WithProfilesWithTransactions['UnregisteredSalesTax']/ReturnsPostY2WithProfilesWithTransactions['OutputTaxBeforeAdjustment']
#ReturnsPostY2WithProfilesWithTransactions['RTaxProp']=ReturnsPostY2WithProfilesWithTransactions['RegisteredSalesTax']/ReturnsPostY2WithProfilesWithTransactions['OutputTaxBeforeAdjustment']

SalesMatch=load_matchsales()
SalesMatch=SalesMatch.drop(['diff','absdiff','maxSalesTax','OtherDeclarationCount','MyDeclarationCount','MatchDeclarationCount','OtherDeclarationTax','MyDeclarationTax','MatchDeclarationTax'],axis=1)
SalesMatch['DealerTIN']=pd.to_numeric(SalesMatch['DealerTIN'],errors='coerce')
SalesMatch['TaxQuarter']=pd.to_numeric(SalesMatch['TaxQuarter'],errors='coerce')

ReturnsPostY2WithProfilesWithTransactionsWithMatch=pd.merge(ReturnsPostY2WithProfiles,SalesMatch, how='left', on=['DealerTIN','TaxQuarter'], indicator='salesmatch_merge')

PurchaseMatch=load_matchpurchases()
PurchaseMatch=PurchaseMatch.drop(['OtherDeclarationCount','MyDeclarationCount', 'MatchDeclarationCount', 'OtherDeclarationTax', 'MyDeclarationTax', 'MatchDeclarationTax', 'diff', 'absdiff', 'maxPurchaseTax'],axis=1)
PurchaseMatch['DealerTIN']=pd.to_numeric(PurchaseMatch['DealerTIN'],errors='coerce')
PurchaseMatch['TaxQuarter']=pd.to_numeric(PurchaseMatch['TaxQuarter'],errors='coerce')

ReturnsPostY2WithProfilesWithTransactionsWithMatch=pd.merge(ReturnsPostY2WithProfilesWithTransactionsWithMatch,PurchaseMatch, how='left', on=['DealerTIN','TaxQuarter'], indicator='purchasematch_merge')

#Importing Network features (sales side)
#Importing Network features (purchase side)
SaleNetworkQuarter=load_salenetwork()
SaleNetworkQuarter['DealerTIN']=pd.to_numeric(SaleNetworkQuarter['DealerTIN'],errors='coerce')
SaleNetworkQuarter['TaxQuarter']=pd.to_numeric(SaleNetworkQuarter['TaxQuarter'],errors='coerce')
FinalEverything=pd.merge(ReturnsPostY2WithProfilesWithTransactionsWithMatch,SaleNetworkQuarter,how='left', on=['DealerTIN','TaxQuarter'], indicator='salesnetwork_merge')

PurchaseNetworkQuarter=load_purchasenetwork()
PurchaseNetworkQuarter['DealerTIN']=pd.to_numeric(PurchaseNetworkQuarter['DealerTIN'],errors='coerce')
PurchaseNetworkQuarter['TaxQuarter']=pd.to_numeric(PurchaseNetworkQuarter['TaxQuarter'],errors='coerce')
FinalEverything=pd.merge(FinalEverything,PurchaseNetworkQuarter,how='left', on=['DealerTIN','TaxQuarter'], indicator='purchasenetwork_merge')

SaleDS=load_salesdownstream()
SaleDS['DealerTIN']=pd.to_numeric(SaleDS['DealerTIN'],errors='coerce')
SaleDS['TaxQuarter']=pd.to_numeric(SaleDS['TaxQuarter'],errors='coerce')
FinalEverything=pd.merge(FinalEverything,SaleDS,how='left', on=['DealerTIN','TaxQuarter'], indicator='salesds_merge')

PurchaseDS=load_purchasedownstream()
PurchaseDS['DealerTIN']=pd.to_numeric(PurchaseDS['DealerTIN'],errors='coerce')
PurchaseDS['TaxQuarter']=pd.to_numeric(PurchaseDS['TaxQuarter'],errors='coerce')
FinalEverything=pd.merge(FinalEverything,PurchaseDS,how='left', on=['DealerTIN','TaxQuarter'], indicator='purchaseds_merge')

FinalEverything_minusq12=FinalEverything[FinalEverything['TaxQuarter']!=12]

del FinalEverything,ReturnsAll,Profiles,ReturnsAllWithProfiles,ReturnsPostY2WithProfiles,ReturnsPostY2WithProfilesWithTransactionsWithMatch
del PurchaseDS, SaleDS, SaleNetworkQuarter, SalesMatch, PurchaseMatch, PurchaseNetworkQuarter

#%%

#ReturnsPostY2WithProfilesWithTransactionsWithMatch=ReturnsPostY2WithProfilesWithTransactionsWithMatch[ReturnsPostY2WithProfilesWithTransactionsWithMatch['TaxQuarter']!=12]
#FramePostY2=load_h2odataframe_returns(ReturnsPostY2WithProfiles)
FrameFinalEverything_minusq12=load_h2odataframe_returns(FinalEverything_minusq12)

TrainData, ValidData, TestData = divide_train_test(FrameFinalEverything_minusq12)
#TrainPostY2, ValidPostY2, TestPostY2 = divide_train_test(FramePostY2)

features=[return_features,dealer_features,all_network_features,return_features+\
          dealer_features,return_features+all_network_features,dealer_features+\
          all_network_features,return_features+dealer_features+all_network_features]

for i in xrange(len(features)):
    rf_models[i].train(features[i], 'bogus_online', training_frame=TrainData, validation_frame=ValidData)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12_newdsfeatures.html',\
                    title='Comparing All Models, Bogus Online')
show(plot)
#%%
for i in xrange(len(rf_models)):
    h2o.save_model(rf_models[i],path='Models')
#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusOnline_model{}_v2.html".format(i+1),n_rows=30)) 

#%%
#generate_predictions(models,ValidationData,FilePath,ColumnTitle):
generate_predictions(rf_models,ValidData,'PredictionsBogusOnline_v2.csv','BogusOnlineModel')
#%%

features=[return_features,dealer_features,all_network_features,return_features+\
          dealer_features,return_features+all_network_features,dealer_features+\
          all_network_features,return_features+dealer_features+all_network_features]

for i in xrange(len(features)):
    rf_models[i].train(features[i], 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusCancellation_comparison_plot_AllCombinations_minusq12.html',\
                    title='Comparing All Models, Bogus Cancellation')
show(plot)
#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusCancellation_model{}_v2.html".format(i+1),n_rows=30)) 

#%%
generate_predictions(rf_models,ValidData,r'PredictionsBogusCancellation_v2.csv','BogusCancellationModel')
#%%
# In this cell, we compare the model where we add match and proportion sales made to registered firms
FinalEverything_minusq12=FinalEverything[FinalEverything['TaxQuarter']!=12]

#ReturnsPostY2WithProfilesWithTransactionsWithMatch=ReturnsPostY2WithProfilesWithTransactionsWithMatch[ReturnsPostY2WithProfilesWithTransactionsWithMatch['TaxQuarter']!=12]
#FramePostY2=load_h2odataframe_returns(ReturnsPostY2WithProfiles)
FrameFinalEverything_minusq12=load_h2odataframe_returns(FinalEverything_minusq12)


FrameFinalEverything_minusq12['TaxQuarter']=FrameFinalEverything_minusq12['TaxQuarter'].asnumeric()

b = FrameFinalEverything_minusq12['TIN_hash_byte']
c = FrameFinalEverything_minusq12['TaxQuarter']
 
train_2012_13 = FrameFinalEverything_minusq12[ (b < 200) & (c < 17)]
valid_2012_13 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c < 17)]
valid_2014 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c >= 17)]

features=[return_features,dealer_features,all_network_features,return_features+\
          dealer_features,return_features+all_network_features,dealer_features+\
          all_network_features,return_features+dealer_features+all_network_features]

for i in xrange(len(features)):
    rf_models[i].train(features[i], 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
   

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12_minusY5.html',\
                    title='Comparing All Models, Bogus Online (Drop Y5 in training)')
show(plot)

#%%
generate_predictions(rf_models,valid_2012_13,'PredictionsBogusOnline_v2_2013_MinusY5.csv','BogusOnlineModel')
generate_predictions(rf_models,valid_2014,'PredictionsBogusOnline_v2_2014_MinusY5.csv','BogusOnlineModel')

#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusOnline_model{}_v2_MinusY5.html".format(i+1),n_rows=30)) 
#%%
features=[return_features,dealer_features,all_network_features,return_features+dealer_features,return_features+all_network_features,dealer_features+all_network_features,return_features+dealer_features+all_network_features]

train_2012 = FrameFinalEverything_minusq12[ (b < 200) & (c < 13)]
valid_2012 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c < 13)]
valid_2013 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c >= 13) & (c < 17)]
valid_2014 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c >= 17)]

for i in xrange(len(features)):
    rf_models[i].train(features[i], 'bogus_online', training_frame=train_2012, validation_frame=valid_2012)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12_minusY5_OnlyY3.html',title='Comparing All Models, Bogus Online (Only Y3 in training)')
show(plot)
#%%
generate_predictions(rf_models,valid_2012,'PredictionsBogusOnline_v2_2012_OnlyY3.csv','BogusOnlineModel')
generate_predictions(rf_models,valid_2013,'PredictionsBogusOnline_v2_2013_OnlyY3.csv','BogusOnlineModel')
generate_predictions(rf_models,valid_2014,'PredictionsBogusOnline_v2_2014_OnlyY3.csv','BogusOnlineModel')
#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusOnline_model{}_v2_OnlyY3.html".format(i+1),n_rows=30, title=legends[i])) 
#%%
#%%
# Lets compare the three bogus variables 
train, valid, test = divide_train_test(fr)

rf_v1 = H2ORandomForestEstimator(
        model_id="rf_v1",
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)

rf_v2 = H2ORandomForestEstimator(
        model_id="rf_v2",
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)

rf_v3 = H2ORandomForestEstimator(
        model_id="rf_v3",
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)

features = ['TurnoverGross', 'MoneyDeposited','VatRatio','LocalVatRatio','TotalReturnCount','RefundClaimedBoolean', 'InterstateRatio', 'MoneyGroup', 'AllCentral', 'AllLocal'] 
  
rf_v1.train(features, 'bogus_online', training_frame=train, validation_frame=valid)
rf_v2.train(features, 'bogus_cancellation', training_frame=train, validation_frame=valid)
rf_v3.train(features, 'bogus_any', training_frame=train, validation_frame=valid)

plot=compare_models(rf_v1,rf_v2, rf_v3, legend1="Bogus from online data", legend2="Bogus from Cancellation data", legend3="Bogus Any")
show(plot)
#%%
ReturnsT15=ReturnsPostY2WithProfilesWithTransactionsWithMatch[ReturnsPostY2WithProfilesWithTransactionsWithMatch['TaxQuarter']==17]

FrameT15=load_h2odataframe_returns(ReturnsT15)

TrainT15, ValidT15, TestT15 = divide_train_test(FrameT15)

# Features that the Tax Authority uses
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross','TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment','TotalReturnCount']
# Features that we think can be important
important_features =  ['PositiveContribution','InterstateRatio','RefundClaimedBoolean', 'MoneyGroup','CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
remaining_features = ['PercPurchaseUnregisteredDealer','PercValueAdded','AllCentral', 'AllLocal', 'ZeroTax', 'ZeroTurnover', 'ZeroTaxCredit']
# Dealer Features
dealer_features=['profile_merge','Ward','Constitution','BooleanRegisteredIEC','BooleanRegisteredCE','BooleanServiceTax','StartYear','DummyManufacturer','DummyRetailer','DummyWholeSaler','DummyInterStateSeller','DummyInterStatePurchaser','DummyWorkContractor','DummyImporter','DummyExporter','DummyOther','DummyHotel','DummyECommerce','DummyTelecom']
# Transaction Features
transaction_features=['transaction_merge','UnTaxProp']
# Match Features
#match_features=['purchasematch_merge','salesmatch_merge', 'PurchasesAvgMatch', 'PurchasesAvgMatch3', 'PurchasesNameAvgMatch','SalesAvgMatch','SalesAvgMatch3','SalesNameAvgMatch' ]
match_features=['purchasematch_merge','salesmatch_merge','SaleMyCountDiscrepancy', 'SaleOtherCountDiscrepancy','SaleMyTaxDiscrepancy','SaleOtherTaxDiscrepancy','SaleDiscrepancy','absSaleDiscrepancy', '_merge_salediscrepancy', 'PurchaseMyCountDiscrepancy','PurchaseOtherCountDiscrepancy','PurchaseMyTaxDiscrepancy','PurchaseOtherTaxDiscrepancy','PurchaseDiscrepancy','absPurchaseDiscrepancy' ]


rf_v1.train(basic_features, 'bogus_online', training_frame=TrainT15, validation_frame=ValidT15)
rf_v2.train(basic_features+important_features+remaining_features, 'bogus_online', training_frame=TrainT15, validation_frame=ValidT15)
rf_v3.train(basic_features+important_features+remaining_features+transaction_features+match_features, 'bogus_online', training_frame=TrainT15, validation_frame=ValidT15)
rf_v4.train(basic_features+dealer_features+important_features+remaining_features, 'bogus_online', training_frame=TrainT15, validation_frame=ValidT15)
rf_v5.train(basic_features+important_features+remaining_features+dealer_features+transaction_features+match_features, 'bogus_online', training_frame=TrainT15, validation_frame=ValidT15)

plot=compare_models(model1=rf_v1,model2=rf_v2,model3=rf_v3,model4=rf_v4,model5=rf_v5, legend1="Basic return features", legend2="Basic and advanced return features", legend3="Basic and advanced return, and network features", legend4="All return features with dealer profiles",legend5="All return features, dealer profiles, and network features", of='Graphs/BogusOnline_comparison_plot_T17_AllCombinations.html',title='Comparing All Models (T17)')
show(plot)

#%%
rf_models = [h2o.load_model('Models/rf_v{}'.format(i+1)) for i in xrange(7)]

for i in xrange(len(rf_models)):
    h2o.save_model(rf_models[i],path='Models')

#%%
"""
My lack of python expertise implies i need to keep certain commands handy

"""

{key:"Sales_"+key for key in ['one','two','blabla','foo']}

# How to open Jupyter from the required directory
jupyter notebook --notebook-dir=E:/Ofir


fr = h2o.import_file(path="E:\data\PreliminaryAnalysis\BogusDealers\FeatureReturns.csv") # r"Z:\sub_returns.csv"

returns["ZeroTaxCredit"]=returns["ZeroTaxCredit"].astype('category')

fr2.show(use_pandas=True)

returns.columns # shows column names
returns.sort_values(by='Columna name')

df['E'][df['E'].isin(['two','four'])]='five'
 
returns.hist(column='ZeroTaxCredit', by='TaxQuarter')
 
 
df.ix[df.AAA >= 5,'BBB'] = -1; df



df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
   ....:                    'size': list('SSMMMLL'),
   ....:                    'weight': [8, 10, 11, 1, 20, 12, 12],
   ....:                    'adult' : [False] * 5 + [True] * 2}); df
   ....: 
Out[89]: 
   adult animal size  weight
0  False    cat    S       8
1  False    dog    S      10
2  False    cat    M      11
3  False   fish    M       1
4  False    dog    M      20
5   True    cat    L      12
6   True    cat    L      12

#List the size of the animals with the highest weight.
In [90]: df.groupby('animal').apply(lambda subf: subf['size'][subf['weight'].idxmax()]) 

h2o.varimp(my_imp.rf)

gb = df.groupby(['animal'])

#%%
# Features that we intuitvely understand 
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross','TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment','TotalReturnCount']
# Features that we think can be important
# We should drop MoneyGroup 'PositiveContribution'
important_features =  ['InterstateRatio','RefundClaimedBoolean', 'CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
# dropping: 'PercValueAdded' 'ZeroTurnover', 'ZeroTaxCredit'
remaining_features = ['PercPurchaseUnregisteredDealer', 'ZeroTax']
# Dealer Features
dealer_features=['profile_merge','Ward','Constitution','BooleanRegisteredIEC','BooleanRegisteredCE','BooleanServiceTax','StartYear','DummyManufacturer','DummyRetailer','DummyWholeSaler','DummyInterStateSeller','DummyInterStatePurchaser','DummyWorkContractor','DummyImporter','DummyExporter','DummyOther','DummyHotel','DummyECommerce','DummyTelecom']
# Transaction Features
transaction_features=['transaction_merge','UnTaxProp']
# Match Features
#match_features=['purchasematch_merge','salesmatch_merge', 'PurchasesAvgMatch', 'PurchasesAvgMatch3', 'PurchasesNameAvgMatch','SalesAvgMatch','SalesAvgMatch3','SalesNameAvgMatch' ]
# Check to see if there is drop in performance when you compare A and D
match_features=['purchasematch_merge','salesmatch_merge','SaleMyCountDiscrepancy', 'SaleOtherCountDiscrepancy','SaleMyTaxDiscrepancy','SaleOtherTaxDiscrepancy','SaleDiscrepancy','absSaleDiscrepancy', '_merge_salediscrepancy', 'PurchaseMyCountDiscrepancy','PurchaseOtherCountDiscrepancy','PurchaseMyTaxDiscrepancy','PurchaseOtherTaxDiscrepancy','PurchaseDiscrepancy','absPurchaseDiscrepancy' ]
network_features=[ u'Purchases_component_id', u'Purchases_pagerank', u'Purchases_triangle_count', u'Purchases_in_degree', u'Purchases_out_degree', u'Purchases_core_id', u'purchasenetwork_merge',u'Sales_component_id', u'Sales_pagerank', u'Sales_triangle_count', u'Sales_in_degree', u'Sales_out_degree', u'Sales_core_id', u'salesnetwork_merge']

# Features that the Tax Authority uses
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross','TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment','TotalReturnCount']
# Features that we think can be important
important_features =  ['PositiveContribution','InterstateRatio','RefundClaimedBoolean', 'MoneyGroup','CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
remaining_features = ['PercPurchaseUnregisteredDealer','PercValueAdded','AllCentral', 'AllLocal', 'ZeroTax', 'ZeroTurnover', 'ZeroTaxCredit']
# Dealer Features
dealer_features=['profile_merge','Ward','Constitution','BooleanRegisteredIEC','BooleanRegisteredCE','BooleanServiceTax','StartYear','DummyManufacturer','DummyRetailer','DummyWholeSaler','DummyInterStateSeller','DummyInterStatePurchaser','DummyWorkContractor','DummyImporter','DummyExporter','DummyOther','DummyHotel','DummyECommerce','DummyTelecom']
# Transaction Features
transaction_features=['transaction_merge','UnTaxProp']
# Match Features
#match_features=['purchasematch_merge','salesmatch_merge', 'PurchasesAvgMatch', 'PurchasesAvgMatch3', 'PurchasesNameAvgMatch','SalesAvgMatch','SalesAvgMatch3','SalesNameAvgMatch' ]
match_features=['purchasematch_merge','salesmatch_merge','SaleMyCountDiscrepancy', 'SaleOtherCountDiscrepancy','SaleMyTaxDiscrepancy','SaleOtherTaxDiscrepancy','SaleDiscrepancy','absSaleDiscrepancy', '_merge_salediscrepancy', 'PurchaseMyCountDiscrepancy','PurchaseOtherCountDiscrepancy','PurchaseMyTaxDiscrepancy','PurchaseOtherTaxDiscrepancy','PurchaseDiscrepancy','absPurchaseDiscrepancy' ]
network_features=[ u'Purchases_component_id', u'Purchases_pagerank', u'Purchases_triangle_count', u'Purchases_in_degree', u'Purchases_out_degree', u'Purchases_core_id', u'purchasenetwork_merge',u'Sales_component_id', u'Sales_pagerank', u'Sales_triangle_count', u'Sales_in_degree', u'Sales_out_degree', u'Sales_core_id', u'salesnetwork_merge']

