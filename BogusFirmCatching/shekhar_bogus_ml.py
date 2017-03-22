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
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross','TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment','TotalReturnCount']
# Features that we think can be important
important_features =  ['PositiveContribution','InterstateRatio','RefundClaimedBoolean', 'MoneyGroup','CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
remaining_features = ['PercPurchaseUnregisteredDealer','PercValueAdded','AllCentral', 'AllLocal', 'ZeroTax', 'ZeroTurnover', 'ZeroTaxCredit','TotalPurchases']

return_features=basic_features+important_features+remaining_features


# Dealer Features
dealer_features=['profile_merge','Ward','Constitution','BooleanRegisteredIEC','BooleanRegisteredCE','BooleanServiceTax','StartYear','DummyManufacturer','DummyRetailer','DummyWholeSaler','DummyInterStateSeller','DummyInterStatePurchaser','DummyWorkContractor','DummyImporter','DummyExporter','DummyOther','DummyHotel','DummyECommerce','DummyTelecom']


# Transaction Features
transaction_features=['transaction_merge','UnTaxProp']
# Match Features
#match_features=['purchasematch_merge','salesmatch_merge', 'PurchasesAvgMatch', 'PurchasesAvgMatch3', 'PurchasesNameAvgMatch','SalesAvgMatch','SalesAvgMatch3','SalesNameAvgMatch' ]
match_features=['purchasematch_merge','salesmatch_merge','SaleMyCountDiscrepancy', 'SaleOtherCountDiscrepancy','SaleMyTaxDiscrepancy','SaleOtherTaxDiscrepancy','SaleDiscrepancy','absSaleDiscrepancy', '_merge_salediscrepancy', 'PurchaseMyCountDiscrepancy','PurchaseOtherCountDiscrepancy','PurchaseMyTaxDiscrepancy','PurchaseOtherTaxDiscrepancy','PurchaseDiscrepancy','absPurchaseDiscrepancy' ]
network_features=[ u'Purchases_pagerank', u'Purchases_triangle_count', u'Purchases_in_degree', u'Purchases_out_degree', u'purchasenetwork_merge', u'Sales_pagerank', u'Sales_triangle_count', u'Sales_in_degree', u'Sales_out_degree', u'salesnetwork_merge']


all_network_features=transaction_features+match_features+network_features
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

# load profiles data
Profiles=load_profile()

#Merge returns data with profile data
ReturnsAllWithProfiles=pd.merge(ReturnsAll, Profiles, how='left', on=['DealerTIN'], indicator='profile_merge')

#save returns only from year 3 onwards (inclusive)
ReturnsPostY2WithProfiles=ReturnsAllWithProfiles[ReturnsAllWithProfiles['TaxQuarter']>8]

# load data related to calculating percentage sales made to registered firms
Transactions=load_registeredsales()

# Merge Returns+Profiles+RegisteredSalesData
ReturnsPostY2WithProfilesWithTransactions = pd.merge(ReturnsPostY2WithProfiles, Transactions, how='left', on=['DealerTIN','TaxQuarter'], indicator='transaction_merge')
ReturnsPostY2WithProfilesWithTransactions['UnTaxProp']=0
ReturnsPostY2WithProfilesWithTransactions['RTaxProp']=0

ReturnsPostY2WithProfilesWithTransactions['UnTaxProp']=ReturnsPostY2WithProfilesWithTransactions['UnregisteredSalesTax']/ReturnsPostY2WithProfilesWithTransactions['OutputTaxBeforeAdjustment']
ReturnsPostY2WithProfilesWithTransactions['RTaxProp']=ReturnsPostY2WithProfilesWithTransactions['RegisteredSalesTax']/ReturnsPostY2WithProfilesWithTransactions['OutputTaxBeforeAdjustment']

SalesMatch=load_matchsales()
SalesMatch=SalesMatch.drop(['diff','absdiff','maxSalesTax','OtherDeclarationCount','MyDeclarationCount','MatchDeclarationCount','OtherDeclarationTax','MyDeclarationTax','MatchDeclarationTax'],axis=1)
ReturnsPostY2WithProfilesWithTransactionsWithMatch=pd.merge(ReturnsPostY2WithProfilesWithTransactions,SalesMatch, how='left', on=['DealerTIN','TaxQuarter'], indicator='salesmatch_merge')

PurchaseMatch=load_matchpurchases()
PurchaseMatch=PurchaseMatch.drop(['OtherDeclarationCount','MyDeclarationCount', 'MatchDeclarationCount', 'OtherDeclarationTax', 'MyDeclarationTax', 'MatchDeclarationTax', 'diff', 'absdiff', 'maxPurchaseTax'],axis=1)
ReturnsPostY2WithProfilesWithTransactionsWithMatch=pd.merge(ReturnsPostY2WithProfilesWithTransactionsWithMatch,PurchaseMatch, how='left', on=['DealerTIN','TaxQuarter'], indicator='purchasematch_merge')

#Importing Network features (sales side)
#Importing Network features (purchase side)
SaleNetworkQuarter=load_salenetwork()
PurchaseNetworkQuarter=load_purchasenetwork()

FinalEverything=pd.merge(ReturnsPostY2WithProfilesWithTransactionsWithMatch,SaleNetworkQuarter,how='left', on=['DealerTIN','TaxQuarter'], indicator='salesnetwork_merge')
FinalEverything=pd.merge(FinalEverything,PurchaseNetworkQuarter,how='left', on=['DealerTIN','TaxQuarter'], indicator='purchasenetwork_merge')

del ReturnsAll,Profiles,ReturnsAllWithProfiles,ReturnsPostY2WithProfiles,Transactions,ReturnsPostY2WithProfilesWithTransactions,ReturnsPostY2WithProfilesWithTransactionsWithMatch

FinalEverything_minusq12=FinalEverything[FinalEverything['TaxQuarter']!=12]
#%%

#ReturnsPostY2WithProfilesWithTransactionsWithMatch=ReturnsPostY2WithProfilesWithTransactionsWithMatch[ReturnsPostY2WithProfilesWithTransactionsWithMatch['TaxQuarter']!=12]
#FramePostY2=load_h2odataframe_returns(ReturnsPostY2WithProfiles)
FrameFinalEverything_minusq12=load_h2odataframe_returns(FinalEverything_minusq12)

TrainData, ValidData, TestData = divide_train_test(FrameFinalEverything_minusq12)
#TrainPostY2, ValidPostY2, TestPostY2 = divide_train_test(FramePostY2)

rf_models[0].train(return_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)
rf_models[1].train(dealer_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)
rf_models[2].train(all_network_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)
rf_models[3].train(return_features+dealer_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)
rf_models[4].train(return_features+all_network_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)
rf_models[5].train(dealer_features+all_network_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)
rf_models[6].train(return_features+dealer_features+all_network_features, 'bogus_online', training_frame=TrainData, validation_frame=ValidData)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12.html',title='Comparing All Models, Bogus Online')
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
rf_models[0].train(return_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_models[1].train(dealer_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_models[2].train(all_network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_models[3].train(return_featuers+dealer_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_models[4].train(return_featuers+all_network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_models[5].train(dealer_features+all_network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_models[6].train(return_features+dealer_features+all_network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusCancellation_comparison_plot_AllCombinations_minusq12.html',title='Comparing All Models, Bogus Cancellation')
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
  


rf_models[0].train(return_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
rf_models[1].train(dealer_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
rf_models[2].train(all_network_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
rf_models[3].train(return_featuers+dealer_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
rf_models[4].train(return_featuers+all_network_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
rf_models[5].train(dealer_features+all_network_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)
rf_models[6].train(return_features+dealer_features+all_network_features, 'bogus_online', training_frame=train_2012_13, validation_frame=valid_2012_13)

legends=["Return features","Profile features","Network features","1 + 2","1 + 3","2 + 3","1 + 2 + 3"]

plot=compare_models(rf_models,legends, of='Graphs/BogusOnline_comparison_plot_AllCombinations_minusq12_minusY5.html',title='Comparing All Models')
show(plot)


#%%
generate_predictions(rf_models,valid_2012_13,'PredictionsBogusOnline_v2_2013_test.csv','BogusOnlineModel')
generate_predictions(rf_models,valid_2014,'PredictionsBogusOnline_v2_2014_test.csv','BogusOnlineModel')

#%%
#%%
PredictionDataModel1=set_predictions(rf_v1,ValidData)
PredictionDataModel1=set_prediction_name(PredictionDataModel1,'p1','BogusOnlineModel1')

PredictionDataModel2=set_predictions(rf_v2,ValidData)
PredictionDataModel2=set_prediction_name(PredictionDataModel2,'p1','BogusOnlineModel2')

PredictionDataModel3=set_predictions(rf_v3,ValidData)
PredictionDataModel3=set_prediction_name(PredictionDataModel3,'p1','BogusOnlineModel3')

PredictionDataModel4=set_predictions(rf_v4,ValidData)
PredictionDataModel4=set_prediction_name(PredictionDataModel4,'p1','BogusOnlineModel4')

PredictionDataModel5=set_predictions(rf_v5,ValidData)
PredictionDataModel5=set_prediction_name(PredictionDataModel5,'p1','BogusOnlineModel5')

PredictionDataModel6=set_predictions(rf_v6,ValidData)
PredictionDataModel6=set_prediction_name(PredictionDataModel6,'p1','BogusOnlineModel6')

PredictionDataModel7=set_predictions(rf_v7,ValidData)
PredictionDataModel7=set_prediction_name(PredictionDataModel7,'p1','BogusOnlineModel7')

Y=pd.concat([PredictionDataModel1,PredictionDataModel2,PredictionDataModel3,PredictionDataModel4,PredictionDataModel5,PredictionDataModel6,PredictionDataModel7],axis=1,ignore_index=False)
#Y=Y.as_data_frame(use_pandas=True)

Z=ValidData.as_data_frame(use_pandas=True)
Z=Z[['DealerTIN','TaxQuarter','bogus_online','bogus_cancellation','profile_merge','transaction_merge','salesmatch_merge','purchasematch_merge','purchasenetwork_merge','salesnetwork_merge']]
Z.index=Z.index.map(unicode)

PredictionData10=pd.concat([Y,Z],axis=1)
PredictionData10.to_csv(path_or_buf='PredictionsBogusOnline_v2.csv')
#%%

rf_v1.train(basic_features+important_features+remaining_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_v2.train(dealer_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_v3.train(transaction_features+match_features+network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_v4.train(basic_features+important_features+remaining_features+dealer_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_v5.train(basic_features+important_features+remaining_features+transaction_features+match_features+network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_v6.train(dealer_features+transaction_features+match_features+network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)
rf_v7.train(basic_features+important_features+remaining_features+dealer_features+transaction_features+match_features+network_features, 'bogus_cancellation', training_frame=TrainData, validation_frame=ValidData)

plot=compare_models(model1=rf_v1,model2=rf_v2,model3=rf_v3,model4=rf_v4,model5=rf_v5,model6=rf_v6,model7=rf_v7, legend1="Return features", legend2="Profile features", legend3="Network features", legend4="1 + 2",legend5="1 + 3",legend6="2 + 3",legend7="1 + 2 + 3", of='Graphs/BogusCancellation_comparison_plot_AllCombinations_minusq12.html',title='Comparing All Models')
show(plot)
#%%
show(analyze_model(rf_v7,of='Graphs/BogusCancellation_model7_v2.html'))
show(analyze_model(rf_v6,of='Graphs/BogusCancellation_model6_v2.html'))
show(analyze_model(rf_v5,of='Graphs/BogusCancellation_model5_v2.html'))
show(analyze_model(rf_v4,of='Graphs/BogusCancellation_model4_v2.html'))
show(analyze_model(rf_v3,of='Graphs/BogusCancellation_model3_v2.html'))
show(analyze_model(rf_v2,of='Graphs/BogusCancellation_model2_v2.html'))
show(analyze_model(rf_v1,of='Graphs/BogusCancellation_model1_v2.html'))

#%%
PredictionDataModel1=set_predictions(rf_v1,ValidData)
PredictionDataModel1=set_prediction_name(PredictionDataModel1,'p1','BogusCancellationModel1')

PredictionDataModel2=set_predictions(rf_v2,ValidData)
PredictionDataModel2=set_prediction_name(PredictionDataModel2,'p1','BogusCancellationModel2')

PredictionDataModel3=set_predictions(rf_v3,ValidData)
PredictionDataModel3=set_prediction_name(PredictionDataModel3,'p1','BogusCancellationModel3')

PredictionDataModel4=set_predictions(rf_v4,ValidData)
PredictionDataModel4=set_prediction_name(PredictionDataModel4,'p1','BogusCancellationModel4')

PredictionDataModel5=set_predictions(rf_v5,ValidData)
PredictionDataModel5=set_prediction_name(PredictionDataModel5,'p1','BogusCancellationModel5')

PredictionDataModel6=set_predictions(rf_v6,ValidData)
PredictionDataModel6=set_prediction_name(PredictionDataModel6,'p1','BogusCancellationModel6')

PredictionDataModel7=set_predictions(rf_v7,ValidData)
PredictionDataModel7=set_prediction_name(PredictionDataModel7,'p1','BogusCancellationModel7')

Y=pd.concat([PredictionDataModel1,PredictionDataModel2,PredictionDataModel3,PredictionDataModel4,PredictionDataModel5,PredictionDataModel6,PredictionDataModel7],axis=1,ignore_index=False)
#Y=Y.as_data_frame(use_pandas=True)

Z=ValidData.as_data_frame(use_pandas=True)
Z=Z[['DealerTIN','TaxQuarter','bogus_online','bogus_cancellation','profile_merge','transaction_merge','salesmatch_merge','purchasematch_merge','purchasenetwork_merge','salesnetwork_merge']]
Z.index=Z.index.map(unicode)

PredictionData10=pd.concat([Y,Z],axis=1)

PredictionData10=pd.concat([PredictionDataModel7,PredictionData10],axis=1)


PredictionData10.to_csv(path_or_buf='PredictionsBogusCancellation_v2.csv')

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
#Comparing features for bogus_online
# Features that the Tax Authority uses
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross','TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment']
# Features that we think can be important
important_features =  ['PositiveContribution','InterstateRatio','TotalReturnCount','RefundClaimedBoolean', 'MoneyGroup','CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
remaining_features = ['PercPurchaseUnregisteredDealer','PercValueAdded','AllCentral', 'AllLocal', 'ZeroTax', 'ZeroTurnover', 'ZeroTaxCredit']

rf_v1.train(basic_features, 'bogus_online', training_frame=train, validation_frame=valid )
rf_v2.train(basic_features+important_features, 'bogus_online', training_frame=train, validation_frame=valid )
rf_v3.train(basic_features+important_features+remaining_features, 'bogus_online', training_frame=train, validation_frame=valid )

plot=compare_models(rf_v1,rf_v2, rf_v3, legend1="Basic Features", legend2="Features that we think are important", legend3="All Features", of='Graphs/comparison_plot_online.html')
show(plot)
#%%
#Comparing features for bogus_cancellation
# Features that the Tax Authority uses
basic_features = ['MoneyDeposited','VatRatio','LocalVatRatio','TurnoverGross','TurnoverLocal','OutputTaxBeforeAdjustment','TaxCreditBeforeAdjustment']
# Features that we think can be important
important_features =  ['PositiveContribution','InterstateRatio','TotalReturnCount','RefundClaimedBoolean', 'MoneyGroup','CreditRatio','LocalCreditRatio']
# Rest of the features in the the returns
remaining_features = ['PercPurchaseUnregisteredDealer','PercValueAdded','AllCentral', 'AllLocal', 'ZeroTax', 'ZeroTurnover', 'ZeroTaxCredit']

rf_v1.train(basic_features, 'bogus_cancellation', training_frame=train, validation_frame=valid )
rf_v2.train(basic_features+important_features, 'bogus_cancellation', training_frame=train, validation_frame=valid )
rf_v3.train(basic_features+important_features+remaining_features, 'bogus_cancellation', training_frame=train, validation_frame=valid )

plot=compare_models(rf_v1,rf_v2, rf_v3, legend1="Basic Features", legend2="Features that we think are important", legend3="All Features", of='Graphs/comparison_plot_cancellation.html')
show(plot)
#%%
# Comparing bogus_online and bogus_cancellation
rf_online_v2 = H2ORandomForestEstimator(
        model_id="rf_online_v2",
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)

rf_cancellation_v2 = H2ORandomForestEstimator(
        model_id="rf_cancellation_v2",
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)

rf_online_v2.train(basic_features+important_features, 'bogus_online', training_frame=train, validation_frame=valid )
rf_cancellation_v2.train(basic_features+important_features, 'bogus_cancellation', training_frame=train, validation_frame=valid )

plot=compare_models(rf_online_v2,rf_cancellation_v2, legend1="bogus_online", legend2="bogus_cancellation", of='Graphs/comparison_plot_onlinevscancellation.html')
show(plot)

#%%
rf_v4.train(basic_features+dealer_features+important_features+remaining_features, 'bogus_cancellation', training_frame=TrainPostY2WithTransactions, validation_frame=ValidPostY2WithTransactions)
rf_v5.train(basic_features+important_features+remaining_features+dealer_features+transaction_features+match_features, 'bogus_cancellation', training_frame=TrainPostY2WithTransactions, validation_frame=ValidPostY2WithTransactions)

plot=compare_models(rf_v1,rf_v4,rf_v5, legend1="Data for last 3 years, with basic return features", legend2="Data for last 3 years, with basic and advanced return features, and dealer profiles", legend3="Last 3 years, with basic and advanced return features, dealer features and network features", of='Graphs/BogusOnline_comparison_plot_PostY2_WithVsWithoutTransaction_withDealerProfiles.html',title='Comparing 3 models, with DP')
show(plot)

plot=compare_models(rf_v1,rf_v4,rf_v3, legend1="Data for last 3 years minus q12, with basic return features", legend2="Data for last 3 years  minus q12, with basic and advanced return features, and dealer profiles", legend3="Last 3 years  minus q12, with basic and advanced return features, and network features", of='Graphs/BogusOnline_comparison_plot_PostY2_WithVsWithoutTransaction_withVswithoutDealerProfiles.html',title='Comparing 3 models, with DP')
show(plot)

show(analyze_model(rf_v5,of='Graphs/allreturnfeatures_withnetworkfeatures_withdealerprofiles_bogusonline_minusq12.html'))

show(analyze_model(rf_v1,of='Graphs/BogusOnline_Model1.html'))
show(analyze_model(rf_v2,of='Graphs/BogusOnline_Model2.html'))
show(analyze_model(rf_v3,of='Graphs/BogusOnline_Model3.html'))
show(analyze_model(rf_v4,of='Graphs/BogusOnline_Model4.html'))
show(analyze_model(rf_v5,of='Graphs/BogusOnline_Model5.html'))
show(analyze_model(rf_v6,of='Graphs/BogusOnline_Model6.html'))
show(analyze_model(rf_v7,of='Graphs/BogusOnline_Model7.html'))

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

