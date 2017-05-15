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
from hashlib import md5

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
    h2o.init(nthreads = -1,max_mem_size="58G")
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
             u'PurchaseDSCreditRatio', u'PurchaseDSVatRatio',u'Missing_MaxPurchaseProp',\
             u'Missing_PurchaseDSUnTaxProp', \
             u'Missing_PurchaseDSCreditRatio', u'Missing_PurchaseDSVatRatio', u'TotalSellers',\
             u'salesds_merge', u'MaxSalesProp',u'Missing_MaxSalesProp', u'SalesDSUnTaxProp',\
             u'SalesDSCreditRatio', u'SalesDSVatRatio', u'Missing_SalesDSUnTaxProp', \
             u'Missing_SalesDSCreditRatio', u'Missing_SalesDSVatRatio', u'TotalBuyers']

all_network_features=transaction_features+match_features+network_features+ds_features

features=[return_features,dealer_features,all_network_features,return_features+\
          dealer_features,return_features+all_network_features,dealer_features+\
          all_network_features,return_features+dealer_features+all_network_features]
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
FinalEverything_minusq12=load_everything()

FrameFinalEverything_minusq12=load_h2odataframe_returns(FinalEverything_minusq12)

#%%
#ReturnsPostY2WithProfilesWithTransactionsWithMatch=ReturnsPostY2WithProfilesWithTransactionsWithMatch[ReturnsPostY2WithProfilesWithTransactionsWithMatch['TaxQuarter']!=12]

TrainData, ValidData, TestData = divide_train_test(FrameFinalEverything_minusq12)
#TrainPostY2, ValidPostY2, TestPostY2 = divide_train_test(FramePostY2)
#all_network_features=all_network_features+['TotalBuyers',u'SalesDSUnTaxProp',u'SalesDSCreditRatio', u'SalesDSVatRatio', u'Missing_SalesDSUnTaxProp', u'Missing_SalesDSCreditRatio', u'Missing_SalesDSVatRatio', u'Missing_MaxSalesProp','salesds_merge']

#features=[return_features,ds_features,return_features+ds_features]

for i in xrange(len(features)):
    print "Building Model {}".format(i+1)
    rf_models[i].train(features[i], 'bogus_online', training_frame=TrainData, validation_frame=ValidData)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]
#legends=["return_features","ds_features","return_features+ds_features"]

plot=compare_models(rf_models,legends, of=r'Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12_numericmerge_withds.html',\
                    title='Comparing All Models, Bogus Online')
show(plot)
#%%
for i in xrange(len(rf_models)):
    h2o.save_model(rf_models[i],path='Models')
#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusOnline_model{}_v2_numericmerge_withds.html".format(i+1),n_rows=30))

#%%
#generate_predictions(models,ValidationData,FilePath,ColumnTitle):
generate_predictions(rf_models,ValidData,'PredictionsBogusOnline_v2_numericmerge_withDS.csv','BogusOnlineModel')
#%%

features=[return_features,dealer_features,all_network_features,return_features+\
          dealer_features,return_features+all_network_features,dealer_features+\
          all_network_features,return_features+dealer_features+all_network_features]

for i in xrange(len(features)):
   print "Building Model {}".format(i+1)
   rf_models[i].train(features[i], 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusCancellation_comparison_plot_AllCombinations_minusq12_numericmerge_withds.html',\
                    title='Comparing All Models, Bogus Cancellation')
show(plot)
#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusCancellation_model{}_v2_numericmerge_withds.html".format(i+1),n_rows=30))

#%%
generate_predictions(rf_models,ValidData,r'PredictionsBogusCancellation_v2_numericmerge_withDS.csv','BogusCancellationModel')
#%%
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

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12_minusY5_withDS.html',\
                    title='Comparing All Models, Bogus Online (Drop Y5 in training)')
show(plot)
#%%
generate_predictions(rf_models,valid_2012_13,'PredictionsBogusOnline_v2_2013_MinusY5_withDS.csv','BogusOnlineModel')
generate_predictions(rf_models,valid_2014,'PredictionsBogusOnline_v2_2014_MinusY5_withDS.csv','BogusOnlineModel')

#%%
for i in xrange(7):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusOnline_v2_MinusY5_withDS_Model{}.html".format(i+1),n_rows=30))
#%%
features=[return_features,dealer_features,all_network_features,return_features+dealer_features,return_features+all_network_features,dealer_features+all_network_features,return_features+dealer_features+all_network_features]

FrameFinalEverything_minusq12['TaxQuarter']=FrameFinalEverything_minusq12['TaxQuarter'].asnumeric()

b = FrameFinalEverything_minusq12['TIN_hash_byte']
c = FrameFinalEverything_minusq12['TaxQuarter']


train_2012 = FrameFinalEverything_minusq12[ (b < 200) & (c < 13)]
valid_2012 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c < 13)]
valid_2013 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c >= 13) & (c < 17)]
valid_2014 = FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & (c >= 17)]

for i in xrange(len(features)):
    print "Building Model {}".format(i+1)
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
#Comparing the two validation sets
rf_models[0].train(features[6], 'bogus_online', training_frame=train_2012, validation_frame=valid_2012)
rf_models[1].train(features[6], 'bogus_online', training_frame=train_2012, validation_frame=valid_2013)

legends=["Validation Set 2012","Validation Set 2013"]

plot=compare_models(rf_models[0:2],legends, of='Graphs/BogusOnline_comparison_plot_Valid2012vsValid2013_minusq12_minusY5_OnlyY3.html',title='Comparing 2 validation sets, Bogus Online (Only Y3 in training)')
show(plot)
#%%
FrameFinalEverything_minusq12['TaxQuarter']=FrameFinalEverything_minusq12['TaxQuarter'].asnumeric()

b = FrameFinalEverything_minusq12['TIN_hash_byte']
c = FrameFinalEverything_minusq12['TaxQuarter']

train_2012 = FrameFinalEverything_minusq12[ (b < 200) & (c < 13)]
valid = {}
valid[0]=FrameFinalEverything_minusq12[(200 <= b) & (b < 232) & (c <=11)]
valid[1]=FrameFinalEverything_minusq12[((200 <= b) & (b < 232)) & ((c==11)|(c==13))]
for i in xrange(13,20):
    valid[i-11]=FrameFinalEverything_minusq12[ (200 <= b) & (b < 232) & ((c==i)|(c==i+1))]

rf_models = [H2ORandomForestEstimator(
        model_id="rf_v{}".format(i),
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000) \
    for i in xrange(1,len(valid)+1)]

for i in xrange(len(valid)):
    rf_models[i].train(features[6], 'bogus_online', training_frame=train_2012, validation_frame=valid[i])

legends=["9-11","11,13","13,14","14,15","15,16","16,17","17,18","18,19","19,20"]

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_SerialValidSets_minusq12_OnlyY3.html',title='Continous validation sets, Only Y3 in training')
show(plot)
#%%
rf_models = [h2o.load_model('Models/BogusOnline_NumericMerge_DS_Model{}'.format(i+1)) for i in xrange(7)]


rf_models = [h2o.load_model('Models/BogusOnline_Model{}_NumericMerge_DS'.format(i+1)) for i in xrange(7)]



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

