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

