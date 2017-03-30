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
    h2o.init(nthreads = -1,max_mem_size="52G")
    h2o.remove_all()

#defining a function to shut down the code
def shutdown_sm():
    h2o.cluster().shutdown()
    %clear
    %reset -f

init()
#%%
# Features that the Tax Authority uses
# Removing 'TurnoverGross','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment',

basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio',\
                  'TurnoverLocal', 'TotalReturnCount']
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


rf_models[4]=H2ORandomForestEstimator(model_id="rf_v5", \
         ntrees=200,stopping_rounds=2, score_each_iteration=True,\
         balance_classes=True,max_after_balance_size=0.8,seed=1000000)
rf_models[5]=H2ORandomForestEstimator(model_id="rf_v6", \
         ntrees=200,stopping_rounds=2, score_each_iteration=True,\
         balance_classes=True,max_after_balance_size=1.6,seed=1000000)



rf_models[6]=H2ORandomForestEstimator(model_id="rf_v7", \
         ntrees=200,stopping_rounds=2, score_each_iteration=True,\
         balance_classes=True,max_after_balance_size=1.6,seed=1000000)
rf_models_7=H2ORandomForestEstimator(model_id="rf_v8", \
         ntrees=200,stopping_rounds=2, score_each_iteration=True,\
         balance_classes=True,max_after_balance_size=1.6,seed=1000000)


#%%
#Merging all data into one file
FinalEverything_minusq12=load_everything()


FinalEverything_minusq12_v2=FinalEverything_minusq12[FinalEverything_minusq12['TurnoverGross']>=0]
FinalEverything_minusq12_v2=FinalEverything_minusq12_v2[FinalEverything_minusq12_v2['TaxCreditBeforeAdjustment']>=0]
FinalEverything_minusq12_v2=FinalEverything_minusq12_v2[FinalEverything_minusq12_v2['OutputTaxBeforeAdjustment']>=0]
del FinalEverything_minusq12

FrameFinalEverything_minusq12=load_h2odataframe_returns(FinalEverything_minusq12_v2)

#%%
#ReturnsPostY2WithProfilesWithTransactionsWithMatch=ReturnsPostY2WithProfilesWithTransactionsWithMatch[ReturnsPostY2WithProfilesWithTransactionsWithMatch['TaxQuarter']!=12]

TrainData, ValidData, TestData = divide_train_test(FrameFinalEverything_minusq12)
#TrainPostY2, ValidPostY2, TestPostY2 = divide_train_test(FramePostY2)
#all_network_features=all_network_features+['TotalBuyers',u'SalesDSUnTaxProp',u'SalesDSCreditRatio', u'SalesDSVatRatio', u'Missing_SalesDSUnTaxProp', u'Missing_SalesDSCreditRatio', u'Missing_SalesDSVatRatio', u'Missing_MaxSalesProp','salesds_merge']

print "Building Model 1"
rf_models[0].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData)
print "Building Model 2"
rf_models[1].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData, weights_column='TurnoverGross')
print "Building Model 3"
rf_models[2].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData, weights_column='OutputTaxBeforeAdjustment')
print "Building Model 4"
rf_models[3].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData, weights_column='TaxCreditBeforeAdjustment')
print "Building Model 5"
rf_models[4].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData)
print "Building Model 6"
rf_models[5].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData)

"""
print "Building Model 7"
rf_models[6].train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData)
print "Building Model 8"
rf_models_7.train(features[6], 'bogus_online', training_frame=TrainData,\
         validation_frame=ValidData, weights_column='TurnoverGross')
"""
legends=["Simple Model","W: Turnover","W: output tax","W: tax credit","simple, balance:0.8","simple, balance:1.6"]
#legends=["return_features","ds_features","return_features+ds_features"]

plot=compare_models(rf_models[:6],legends, of=r'Graphs/BogusOnline_comparison_plot_DifferentRFModels.html',\
                    title='Comparing different RF Models, Bogus Online')
show(plot)

#%%
generate_predictions(rf_models[:6],ValidData,'PredictionsBogusOnline_DifferentModels.csv','BogusWeightsModel')

#%%
for i in xrange(len(rf_models[:6])):
    h2o.save_model(rf_models[i],path='Models/rf_comparison')

#%%
for i in xrange(6):
    show(analyze_model(rf_models[i],of=r"Graphs/BogusOnline_ComparsionRF_Model{}.html".format(i+1),n_rows=30)) 

#%%

FinalEverything_minusq12['TDummy']=FinalEverything_minusq12['TurnoverGross']<0
FinalEverything_minusq12['OTDummy']=FinalEverything_minusq12['OutputTaxBeforeAdjustment']<0
FinalEverything_minusq12['TCDummy']=FinalEverything_minusq12['TaxCreditBeforeAdjustment']<0
