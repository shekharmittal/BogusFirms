os.chdir('E:\Ofir')
from init import *

from art import *
ef('E:\Ofir\init.py')
# from tqdm import tqdm
# tqdm.pandas(desc="my bar!")


columns_2a2b = ['SaleOrPurchase', 'SalePurchaseType', 'DealerGoodType','Rate', 'Amount', 'TaxAmount', 'TotalAmount','SellerBuyerTIN', 'DealerTIN', 'SourceFile']

def load_annexures_2a2b(nrows=None):
    twoa13 = load_annexures_2a2b_2013(None)
    twoa14 = load_annexures_2a2b_2014(None)
    # twoa14 = pd.read_pickle(r'z:\twoa14.pickle')
    # twoa14.to_pickle(r'z:\twoa14.pickle')
    return twoa13,twoa14

def load_annexures_2a2b_2014(nrows=None):
    #twoa14 = pd.read_stata(r"E:\data\annexure_2A2B_quarterly_2014.dta")
    twoa14 = pd.read_csv(r"E:\data\annexure_2A2B_quarterly_2014.csv",nrows=nrows,usecols=columns_2a2b)
    twoa14 = twoa14.query('SaleOrPurchase=="AE" | SaleOrPurchase=="AN" | SaleOrPurchase=="BF"').reset_index() # excluding 78 observations
    twoa14['SellerBuyerTIN'] = pd.to_numeric(twoa14.SellerBuyerTIN,errors='coerce')
    twoa14['DealerGoodType'].loc[twoa14['DealerGoodType']=='uD']='UD'
    twoa14['Year'] = 2014
    source_file2quarter = {'t9854{}14'.format(i):i for i in xrange(1,5)}
    twoa14['Quarter'] = twoa13['SourceFile'].apply(source_file2quarter.get)
    # clean Rate
    try:
        twoa14['Rate'].loc[(twoa14.Rate=='Not Required')] = '0'
    except:
        pass
    twoa14['Rate'] = pd.to_numeric(twoa14['Rate'])
    twoa14['Rate'].loc[twoa14['Rate']==4.99]=5
    return twoa14
    # twoa14.pivot_table('Amount','SourceFile','Quarter',aggfunc=len,fill_value=0)

def load_annexures_2a2b_2013(nrows=None):
    twoa13 = pd.read_csv(r"E:\data\annexure_2A2B_quarterly_2013.csv",nrows=nrows,usecols=columns_2a2b)
    twoa13 = twoa13.query('SaleOrPurchase=="AE" | SaleOrPurchase=="AN" | SaleOrPurchase=="BF"').reset_index() # excluding 24 observations
    twoa13['SellerBuyerTIN'] = pd.to_numeric(twoa13.SellerBuyerTIN,errors='coerce')
    twoa13['DealerGoodType'] = twoa13['DealerGoodType'].apply(lambda x:x.rstrip().upper() if pd.notnull(x) else x)
    twoa13['Year'] = 2013
    source_file2quarter = {'t9854{}13'.format(i):i for i in xrange(1,5)}
    twoa13['Quarter'] = twoa13['SourceFile'].apply(source_file2quarter.get)

    # clean Rate
    try:
        twoa13['Rate'].loc[(twoa13.Rate=='Not Required')] = '0'
    except:
        pass
    # twoa13['Rate'] = twoa13['Rate'].apply(lambda rate: rate.replace(',','.'))
    twoa13['Rate'] = pd.to_numeric(twoa13['Rate'],errors='coerce')
    return twoa13

def investigate_non_applicable_rates(twoa14):
    """ investigate non-standard rates. Not in local sales. """
    rate3 = twoa14[twoa14['Rate']==3]
    m = pd.merge(rate3,twoa14.loc[twoa14['SaleOrPurchase']!='BF',['DealerTIN','SellerBuyerTIN','Rate','Amount','TaxAmount','TotalAmount']],'left',left_on=['DealerTIN','SellerBuyerTIN','Amount'],right_on=['SellerBuyerTIN','DealerTIN','Amount'])
    m.dropna(subset=['Rate_y'])[['Rate_x','Rate_y','Amount','TaxAmount_x','TaxAmount_y','TotalAmount_x','TotalAmount_y']]
    # only 14 matched, and they have different rates and tax amounts
    rate_cleaning = {3.0:nan, 1.0:1, 2.0:20, 20.0:20, 4.0:nan, 0.1:1, 2.5:nan, 6.0:nan, 15.5:nan, 0.125:12.5, 0.05:5, 13.0:nan, 4.99:5, 1.5:nan}
    # twoa14['Rate'][twoa14['Rate']<1] *= 100
    twoa14['Rate'] = twoa14['Rate'].apply(rate_cleaning.get)

def explore_sales_duplicates(sales):
    dd = sales.duplicated(['SourceFile', 'DealerTIN', 'SellerBuyerTIN', 'Rate', 'Amount', 'TaxAmount'],keep=False)
    dd2 = sales.duplicated(['SourceFile', 'DealerTIN', 'SellerBuyerTIN', 'Rate', 'Amount', 'TaxAmount','DateTime','TransactionType','DealerGoodType'],keep=False)
    dd.sum()
    (dd2 != dd).sum()

def descriptives_2a2b(twoa):
    len(twoa)
    # 24228042
    len(twoa['DealerTIN'].unique())
    # 247294
    twoa['SellerBuyerTIN'].isnull().sum()
    # 4423180
    twoa['SellerBuyerTIN'] = pd.to_numeric(twoa.SellerBuyerTIN,errors='coerce')
    # a few more are coerced to nan, various issues.
    len(twoa.drop_duplicates(['DealerTIN','SellerBuyerTIN']))
    # 7814530
    pd.crosstab(twoa.SaleOrPurchase,twoa.SalePurchaseType) # sub-category
    pd.crosstab(twoa.SaleOrPurchase,twoa.Quarter,twoa.TotalAmount,aggfunc=np.sum)

    twoa.columns
    # Index([u'SaleOrPurchase', u'SalePurchaseType', u'DealerGoodType',
    #        u'TransactionType', u'Rate', u'Amount', u'TaxAmount', u'TotalAmount',
    #        u'SellerBuyerTIN', u'DealerName', u'DealerTIN', u'Month', u'Year',
    #        u'DateTime', u'ReceiptId', u'Date', u'FormsStatus', u'TaxRate',
    #        u'AEBoolean', u'T985DF1', u'T985DF2', u'T985DF3', u'T985DF4',
    #        u'T985DF5', u'T985DF6', u'T985DF7', u'T985DF8', u'SourceFile',
    #        u'Quarter'],
    #       dtype='object')

    # generally TotalAmount = Amount + TaxAmount,
    sav = twoa[[u'Amount', u'TaxAmount', u'TotalAmount']][abs(twoa['TotalAmount'] - twoa['TaxAmount'] - twoa['Amount'])>1]
    # but sometimes (4414201 cases) TotalAmount=0 inexplicably and the equality fails
    (sav['TotalAmount']>0).sum()
    # in rare cases (398) the equality fails but TotalAmount != 0
    twoa[[u'Amount', u'TaxAmount', u'TotalAmount']][abs(twoa['TotalAmount'] - twoa['TaxAmount'] - twoa['Amount'])>1]

    aggs = twoa.groupby(['DealerTIN','SaleOrPurchase'])[[u'Amount', u'TaxAmount', u'TotalAmount']].sum().reset_index().fillna(0)
    ct = pd.crosstab(aggs.DealerTIN,aggs.SaleOrPurchase,aggs.TotalAmount, aggfunc=np.sum)
    aggs_ct = ct.reset_index()
    dealer_returns_14 = sr[sr.Year==2014].groupby('DealerTIN')[['TurnoverGross','AmountDepositedByDealer']].sum().reset_index()
    m = pd.merge(aggs_ct[:10000],dealer_returns_14[:10000], 'inner','DealerTIN')
    plt.loglog((m['BF']),(m['TurnoverGross']),'.')
    plt.loglog([1,exp(25)],[1,exp(25)],'-')
    plt.xlabel('BF TotalAmount (log scale)'); plt.ylabel('TurnoverGross (log scale)')
    plt.show()
    m2 = m.dropna()
    mod = sm.OLS(m2['TurnoverGross'], m2[['AE','AN','BF']])
    res = mod.fit()
    print res.summary()

def explore_matching_sales_purchases(twoa):
    # mismatching example
    view_flux(twoa,330718,444421,True,4)
    # matching example with different numbers of transactions
    view_flux(twoa,398379,307317,True,4)

    gr = twoa.groupby(['DealerTIN','Year','Quarter','Rate','SellerBuyerTIN','SaleOrPurchase'])
    red = gr.first()
    red[['Amount','TaxAmount','TotalAmount']] = gr[['Amount','TaxAmount','TotalAmount']].sum()
    red.reset_index(inplace=True)
    keep_cols = ['SaleOrPurchase', 'SalePurchaseType', 'Rate', 'Amount', 'TaxAmount', 'TotalAmount', 'SellerBuyerTIN', 'DealerTIN', 'Year','Quarter']
    sub = red[keep_cols]

    # match sales (left) to purchases (right)
    mm = pd.merge(sub.query('SaleOrPurchase=="BF" & SalePurchaseType=="LS"'),\
        sub.query('SaleOrPurchase!="BF"'),'inner',\
        left_on=['DealerTIN','SellerBuyerTIN','Rate','Quarter'],\
        right_on=['SellerBuyerTIN','DealerTIN','Rate','Quarter'])
    # now compare and see if the totals match. <><><>
    mm[['DealerTIN','SellerBuyerTIN','TotalAmount_x','TotalAmount_y']][:10]

def view_flux(twoa,tin1,tin2,group_by_rate=False,quarter=None):
    """
    @group_by_rate - aggregate txns with the same rate to one row
    @quarter = None - the whole year
    """
    quarter_selection = (twoa.Quarter==quarter) if quarter is not None else True
    display_columns = [u'DealerTIN', u'SellerBuyerTIN', u'SaleOrPurchase',u'Quarter', u'Rate', u'Amount', u'TaxAmount', u'TotalAmount', u'DealerGoodType',
       u'TransactionType',  u'Month', u'Year', u'SalePurchaseType',
       u'Date' ]
    for tin_a,tin_b in [(tin1,tin2),(tin2,tin1)]:
        df = twoa[display_columns][(twoa.DealerTIN==tin_a) & (twoa.SellerBuyerTIN==tin_b) & quarter_selection]
        if group_by_rate:
            gr = df.groupby(['SaleOrPurchase','Rate','Quarter'])
            df = gr.first()
            df[['Amount','TaxAmount','TotalAmount']] = gr[['Amount','TaxAmount','TotalAmount']].sum()
        print df
        print '----------------------------------------------'

def share_to_unregistered(sales,variable,name_in_output,features2a2b=None,aggfunc=np.sum):
    """
    @features2a2b - if not None, results will be added to the existing table which is already indexed by DealerTIN,Year,Quarter
    """
    # Share of tax amount to unregistered
    tab = pd.pivot_table(sales, variable, ['DealerTIN','Year','Quarter'],['DealerGoodType'],aggfunc=aggfunc,fill_value=0)
    if 'UD' not in tab.columns:
        tab['UD'] = 0
    # only include RD,UD, but not sure this is advisable
    tab.drop([c for c in tab.columns if c not in ['RD','UD']],axis=1,inplace=True)
    if features2a2b is None:
        features2a2b = tab
    features2a2b['total_{}'.format(name_in_output)] = tab.sum(axis=1)
    # todo: maybe use ordinary division?
    features2a2b['share_{}_ud'.format(name_in_output)] = div( tab['UD'], features2a2b['total_{}'.format(name_in_output)])
    if 'RD' in features2a2b.columns:
        features2a2b.drop(['RD','UD'], axis=1, inplace=True)
    return features2a2b


# TODO: replace this with entropy?
def share_to_largest(sales):
    gr3 = sales.groupby(['DealerTIN','Year','Quarter','SellerBuyerTIN'])['Amount'].sum().\
        reset_index(level=-1).groupby(level=(0,1,2))
    return gr3['Amount'].max() / gr3['Amount'].sum()
    # return gr3.apply(lambda x: x['Amount'].max()/x['Amount'].sum())

def share_to_largest2(sales):
    sales.sort_values('DealerTIN',inplace=True)
    tins = sales['DealerTIN'].as_matrix()
    switches = where(delta_sub(tins)!=0)[0]+1
    sub_switches = [0]+switches[50::100].tolist()+[len(tins)]
    dfs = []
    for i in tqdm(lrange(sub_switches[:-1])):
        dealer_sales = sales[sub_switches[i]:sub_switches[i+1]]
        gr3 = dealer_sales.groupby(['DealerTIN','Year','Quarter','SellerBuyerTIN'])['Amount'].sum().\
            reset_index(level=-1).groupby(level=(0,1,2))
        dfs.append( gr3['Amount'].max() / gr3['Amount'].sum() )

    return pd.concat(dfs)
    # return gr3.apply(lambda x: x['Amount'].max()/x['Amount'].sum())


def extract_2a2b_features(twoa,export=False,sanity_check=False):
    """ twoa = load_annexures_2a2b() """
     # local sales only. See abbreviations spreadsheet.
    sales = twoa.query('SaleOrPurchase=="BF" & SalePurchaseType=="LS"')
    # reclassify sales where the buyer TIN is null as Unregistered
    sales.loc[(sales['DealerGoodType']=='RD') & sales['SellerBuyerTIN'].isnull(),'DealerGoodType']='UD'

    # Share to unregistered firms features.
    # Share of sales amount to unregistered
    features2a2b = share_to_unregistered(sales,'Amount','sale_amt')
    # Share of tax amount to unregistered
    features2a2b = share_to_unregistered(sales,'TaxAmount','sale_tax_amt',features2a2b)
    # Share of transactions to unregistered
    features2a2b = share_to_unregistered(sales,'SellerBuyerTIN','sale_txns',features2a2b,aggfunc=len) # todo: maybe len(unique())
    # share of purchases by largest buyer
    features2a2b['largest_buyer_amt_share'] = share_to_largest(sales)

    gr = sales.groupby(['DealerTIN','Year','Quarter'])
    features2a2b['n_buyers'] = gr['SellerBuyerTIN'].unique().apply(len)

    # # PURCHASES - inexplicable bug kills python
    # # *********
    # purchases = twoa.query('SaleOrPurchase!="BF"')

    # # share of sales by largest supplier
    # features2a2b['largest_supplier_amt_share'] = share_to_largest(purchases)
    # gr = purchases.groupby(['DealerTIN','Year','Quarter'])
    # features2a2b['n_sellers'] = gr['SellerBuyerTIN'].unique().apply(len)


    if export:
        # export
        features2a2b.reset_index().to_csv(r"F:\BogusDealer_analysis\features_from_2a2b_v2.csv",index=False)

    if sanity_check:
        # sanity check
        tf = tag_returns(features2a2b.reset_index())
        attr_cumulative(tf[tf.total_sale_amt>0],'share_sale_amt_ud')
        attr_cumulative(tf[tf.total_sale_amt>0],'total_sale_amt','bogus_cancellation')

    return features2a2b

def check_python_kill():
    purchases = twoa14.query('SaleOrPurchase!="BF"')
    gr3 = purchases.groupby(['DealerTIN','Year','Quarter','SellerBuyerTIN'])
    print 1,
    # this next one kills python
    gr3 = purchases.groupby(['DealerTIN','Year','Quarter','SellerBuyerTIN'])['Amount'].sum()
    print 2,
    gr3 = purchases.groupby(['DealerTIN','Year','Quarter','SellerBuyerTIN'])['Amount'].sum().reset_index(level=-1)
    print 3,
    gr3 = purchases.groupby(['DealerTIN','Year','Quarter','SellerBuyerTIN'])['Amount'].sum().reset_index(level=-1).groupby(level=(0,1,2))
    print 4,

    features2a2b = extract_2a2b_features(twoa14, True)
    print 'done 2014'
    twoa13 = load_annexures_2a2b_2013()
    features2a2b = extract_2a2b_features(twoa13, True)

import networkx as nx
def fancy_network_stuff(twoa13,twoa14):
    twoa = twoa14
    sales = twoa.query('Quarter==4 & SaleOrPurchase=="BF" & SalePurchaseType=="LS"')
    edges = sales.query('Amount!=0').groupby(['DealerTIN','SellerBuyerTIN'])['Amount'].sum().to_frame().reset_index()
    len(edges)
    # 2765215 - for all 2014
    len(edges.DealerTIN.unique())
    #  169199 - for all 2014

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges.apply(lambda x: (x.DealerTIN,x.SellerBuyerTIN,x.Amount),axis=1))
    G.add_weighted_edges_from(edges[['DealerTIN','SellerBuyerTIN','Amount']].as_matrix())
    # G.edges(data=True)
    # nx.neighbors(G,1344)

    te = tag_returns(edges)
    te = tag_returns(te,'SellerBuyerTIN')
    edges.filter(regex='bogus').corr().to_csv(r'z:\corr.csv')

    res = edges.groupby('DealerTIN').apply(lambda x: np.average(x.buyer_bogus_any, weights=x.Amount))
    fr = res.to_frame('buyer_bogus_any')
    fr['dealer_bogus_any'] = edges.groupby('DealerTIN')['bogus_any'].first()
    fr.reset_index(inplace=True)
    fr[['dealer_bogus_any','buyer_bogus_any']].corr()

    # interesting to investigate the distribution - do some companies buy from many bogus dealers?
    sns.distplot(fr['buyer_bogus_any']) # matplotlib failed, killed python
