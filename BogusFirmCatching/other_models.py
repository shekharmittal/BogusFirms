# -*- coding: utf-8 -*-

#%%
# Initiatlising the file
execfile(r'init_sm.py')
execfile(r'Graphs\graphsetup.py')
execfile(r'ml_funcs.py')

import statsmodels.api as sm
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
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
#TrainData=FinalEverything_minusq12[FinalEverything_minusq12['TIN_hash_byte']<232]
TrainData=FinalEverything_minusq12[FinalEverything_minusq12['TIN_hash_byte']<200]
ValidData=FinalEverything_minusq12[(FinalEverything_minusq12['TIN_hash_byte']<232)&(FinalEverything_minusq12['TIN_hash_byte']>=200)]
TrainData['FoldValue']=TrainData['TIN_hash_byte']%8

FrameTrainData=load_h2odataframe_returns(TrainData)
FrameValidData=load_h2odataframe_returns(ValidData)

rf_cv_model=H2ORandomForestEstimator(model_id="rf_cv", nfolds=7,fold_column='FoldValue',keep_cross_validation_predictions=True, ntrees=200, stopping_rounds=2, score_each_iteration=True,seed=1000000)
#rf_cv_model=H2ORandomForestEstimator(model_id="rf_cv",  ntrees=200, stopping_rounds=2, score_each_iteration=True,seed=1000000)
#nfolds=7,fold_column='FoldValue',keep_cross_validation_predictions=True
rf_cv_model.train(features[6], 'bogus_online', training_frame=FrameTrainData, validation_frame=FrameValidData)
show(analyze_model(rf_cv_model,of=r"BogusFirmCatching/Graphs/BogusOnline_model_v2_numericmerge_withds_cv.html",n_rows=30))
#%%
h2o.save_model(rf_cv_model,path='Models')

#%%
generate_predictions(rf_cv_model,FrameValidData,,'BogusOnlineModelCV')

X=set_predictions(rf_cv_model,FrameValidData)
X=set_prediction_name(X,'p1','BogusOnlineModelCVModel7')
Z=FrameValidData.as_data_frame(use_pandas=True)
Z=Z[['DealerTIN','TaxQuarter','bogus_online','bogus_cancellation','profile_merge',\
         'transaction_merge','salesmatch_merge','purchasematch_merge','purchasenetwork_merge',\
         'salesnetwork_merge']]
Z.index=Z.index.map(unicode)

PredictionData=pd.concat([Z,X],axis=1,ignore_index=False)
PredictionData.to_csv(path_or_buf='PredictionsBogusOnline_cv.csv')# -*- coding: utf-8 -*-
#%%
gbm_v1 = H2OGradientBoostingEstimator(\
    model_id="gbm_v1", ntrees=200, stopping_rounds=3, score_each_iteration=True,seed=1000000,balance_classes=True)
gbm_v1.train(features[6], 'bogus_online', training_frame=FrameTrainData, validation_frame=FrameValidData)
show(analyze_model(gbm_v1,of=r"BogusFirmCatching/Graphs/BogusOnline_model_gbm.html",n_rows=30))

#%%
gbm_v2 = H2OGradientBoostingEstimator(\
    model_id="gbm_v2", ntrees=200, stopping_rounds=2, score_each_iteration=True,seed=1000000,balance_classes=True,max_after_balance_size=0.8)
gbm_v2.train(features[6], 'bogus_online', training_frame=FrameTrainData, validation_frame=FrameValidData)
show(analyze_model(gbm_v2,of=r"BogusFirmCatching/Graphs/BogusOnline_model_gbm_v2.html",n_rows=30))

#%%
gbm_v3 = H2OGradientBoostingEstimator(\
    model_id="gbm_v3", ntrees=200, stopping_rounds=2, score_each_iteration=True,seed=1000000,balance_classes=True,max_after_balance_size=0.8)
gbm_v3.train(features[4], 'bogus_online', training_frame=FrameTrainData, validation_frame=FrameValidData)
show(analyze_model(gbm_v3,of=r"BogusFirmCatching/Graphs/BogusOnline_model_gbm_v3.html",n_rows=30))


h2o.save_model(gbm_v1,path='Models')
h2o.save_model(gbm_v2,path='Models')
h2o.save_model(gbm_v3,path='Models')


#%%
# Gridsearch

## Depth 10 is usually plenty of depth for most datasets, but you never know
hyper_params_tune = {'max_depth':[5,10,15,20,25,30],'ntrees':[10,50,100,200,500],'stopping_rounds':[2,3,4]} ##faster for larger datasets
search_criteria_tune = {'strategy': "Cartesian"}

                        #'seed':1000000}

TrainData, ValidData, TestData = divide_train_test(FrameFinalEverything_minusq12)

grid_search=H2OGridSearch(H2ORandomForestEstimator,  grid_id = 'rf_grid', hyper_params=hyper_params_tune, search_criteria=search_criteria_tune)
grid_search.train(features[6], 'bogus_online', training_frame=TrainData, validation_frame=ValidData)

h2o.save_model(grid_search,path='Models')

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(grid_search[:7],legends, of='Graphs/BogusCancellation_comparison_plot_AllCombinations_minusq12_numericmerge_withds.html',\
                    title='Comparing All Models, Bogus Cancellation')
show(plot)

## Sort the grid models by AUC
sorted_final_grid = grid_search.get_grid(sort_by='tpr',decreasing=True)

print sorted_final_grid


for i in xrange(len(grid_search)):
    h2o.save_model(grid_search[i],path='Models/GridSearch')


import sys

orig_stdout = sys.stdout
f = open('Models/GridSearch/grids.txt', 'w')
sys.stdout = f

print grid_search

sys.stdout = orig_stdout
f.close()


