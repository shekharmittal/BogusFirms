from art import *
ef(r'D:\Ofir\ofir_ml_funcs.py')
ef(r'D:\Ofir\init.py')
#%%
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
import statsmodels.api as sm
#%%
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

    h2o.init()
    h2o.remove_all()

    sr = read_sub_returns()
    fr,share_cols = prepare_h2o_frame()

def read_sub_returns():
    # returns.to_csv(r"Z:\tmp.csv",index=False)
    # returns[['VatRatio','LocalVatRatio','TurnoverGross','bogus_any']].to_csv(r"Z:\returns_tagged_subset.csv",index=False)
    sr = pd.read_csv(r"H:\data\returns\sub_returns_no_strings.csv")
    sr = sr[ sr['Year']>=2013 ]
    return sr

def descriptives():
    len(sr[sr.bogus_online].DealerTIN.unique())
    # 611
    len(sr[sr.bogus_cancellation].DealerTIN.unique())
    # 6968
    len(sr[sr.bogus_any].DealerTIN.unique())
    # 7271
    # out of 336167 total dealers


def prepare_h2o_frame():
    fr = h2o.import_file(path=r"H:\data\returns\sub_returns_no_strings.csv") # r"Z:\sub_returns.csv"
    fr = fr[fr['Year'] >=2013]
    fr['y'] = fr['bogus_online'].asfactor() # bogus_any
    fr['y2']= fr['bogus_online'].asfactor()

    share_cols = []
    for column in ['TurnoverCentral', 'TurnoverLocal', 'TurnoverAt1', 'TurnoverAt5', 'TurnoverAt125', 'TurnoverAt20', 'WCTurnoverAt5', 'WCTurnoverAt125','RefundClaimed','TaxCreditBeforeAdjustment','BalanceCarriedNextTaxPeriod','AdjustCSTLiability']:
        new_col = column+'_share'
        share_cols.append(new_col)
        fr[new_col] = div(fr[column],fr['TurnoverGross'])

    return fr,share_cols

def divide_train_test(fr):
    """ train and test sets - outdated. In future - divide by DealerTIN """
    b = fr['TIN_hash_byte']
    train = fr[ b < 200 ]
    valid = fr[ (200 <= b) & (b < 232) ]
    test  = fr[ 232 <= b ]
    return train, valid, test

def toy_classifications():
    # train, valid, test = fr.split_frame([0.6, 0.2], seed=1234) # simply subsets of fr
    train, valid, test = divide_train_test(fr)


    m = H2OGeneralizedLinearEstimator(family="binomial")
    features = ['VatRatio','LocalVatRatio','TurnoverGross','TotalReturnCount','RefundClaimedBoolean'] + share_cols
    m.train(x=features, y="y", training_frame=train)
    m.confusion_matrix()
    # or m.model_performance() or simply m


    # m = H2ODeepLearningEstimator()
    m.train(x=features, y="y", training_frame=train, validation_frame=valid)
    m.confusion_matrix(valid=True)
    plt.plot(*m.roc(valid=1))
    # m.model_performance(test_data=test)

    # Random Forest
    var_y = 'y'
    rf_v1 = H2ORandomForestEstimator(
        model_id="rf_v1",
        ntrees=200,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)

    rf_v1.train(features, var_y, training_frame=train, validation_frame=valid)
    rf_v1.confusion_matrix(valid=1)
    # plt.plot(*rf_v1.roc(valid=1))
    plot_betas(rf_v1.roc(valid=1))

def sanity_check_perfect_prediction():
    # with the class as a feature
    rf_v1.train(features+['y2'], var_y, training_frame=train, validation_frame=valid)
    rf_v1.confusion_matrix(valid=1) # no errors, perfect performance as expected
    plot_betas(rf_v1.roc(valid=1))

def model_improvement_check(model,train,valid,features,added_features,more_features=None):
    roc_list = []
    for model_features in [features, features+added_features] + ([features+added_features+more_features] if more_features is not None else []):
        model.train(model_features, 'y', training_frame=train, validation_frame=valid)
        roc_list.append( model.roc(valid=1) )

    labels = ['base','added features','added + more features']
    for i,roc in enumerate(roc_list):
        plot_betas(roc,label=labels[i])
    plt.legend()
    plt.show()

def yearly_returns_count(sr):
    gr = sr.dropna(subset=turnover_columns).groupby(['DealerTIN','Year']) # better with "returns"
    sz = gr.size()
    print sz.reset_index(name='size').groupby(['Year','size']).size()
    # 2010-2012: <=4 or 12 predominantly
    # 2012: predominantly 4, some 12
    # 2013-2014: predominantly 4, never >4 (in 2013 5 is very rare but exists)

def check_volatility(sr,time_unit='quarter'):
    """
    Will create and check predictivity of turnover volatility measures
    @sr - sub-returns or returns.
    @time_unit is "quarter" or "year" (case-insensitive)
    """

    if time_unit.lower()=='year':
        time_columns = ["Year"]
    elif time_unit.lower()=='quarter':
        time_columns = ["Year","Quarter"]
    else:
        raise ValueError('invalid time unit {}. Must be "quarter" or "year"'.format(time_unit))
    turnover_columns = ['TurnoverCentral','TurnoverGross','TurnoverLocal','AmountDepositedByDealer','TaxCreditBeforeAdjustment']
    gr = sr.dropna(subset=turnover_columns+time_columns).groupby(['DealerTIN']+time_columns)
    dealer_time_level = gr.first()
    dealer_time_level[turnover_columns] = gr[turnover_columns].mean() #.sum() # If we used older years, summing turnovers for each year to have constant benchmark
    dealer_level = dealer_time_level.reset_index().drop_duplicates('DealerTIN')
    for c in turnover_columns:
        dealer_level[c+'_rsd'] = div(dealer_time_level[c].std(level=0),dealer_time_level[c].mean(level=0))
    for c in turnover_columns:
        attr_cumulative(dealer_level,c+'_rsd')

    # these seem to give almost no predictive power

    # Now I can add the results from "red" to the returns DB - but careful, it uses the future.

def turnover_distribution_by_year_plot(dealer_year_level):
    """
    this is to see that I should really sum the returns from each year.
    If I do mean instead of sum for dealer_year_level I get 2012-2014 systematically lower
    """
    aa = dealer_year_level.reset_index()
    for year in sorted(aa.Year.unique()):
        sns.distplot(np.log1p(aa[aa.Year==year]['TurnoverGross']).dropna(),label=str(year))

    # and excluding 0 turnovers:
    for year in sorted(aa.Year.unique()):
        sns.distplot(np.log1p(aa[(aa['TurnoverGross']>0) & (aa.Year==year)]['TurnoverGross']).dropna(),label=str(year),hist=False)

