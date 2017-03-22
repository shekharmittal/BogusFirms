import sys
from art import *
# ef(r'D:\Ofir\ofir_bayes.py')
import scipy
import scipy.optimize
from numpy import *
import time
POINT_SIZE = 2
_epsilon = 1e-7

def stretch(lst,ratio=1.1):
    arr = array(lst)
    arr = mean(arr) + ( arr - mean(arr) )*ratio
    return arr.tolist()

def stretch_plot(ratio=1.1):
    xl = plt.xlim()
    plt.xlim(stretch(xl,ratio))
    yl = plt.ylim()
    plt.ylim(stretch(yl,ratio))

def attr_cumulative(T,field,by_field='bogus_online',uniform=True,same_figure=False,sum_to_one=True,threaded=False,show_coords=True):
    if (not threaded):
        attr_cumulative_plot(T=T, field=field, by_field=by_field, uniform=uniform, same_figure=same_figure, sum_to_one=sum_to_one, show_coords=show_coords)
    else:
        thread_plot = threading.Thread(target=attr_cumulative_plot, args=(T, field, by_field, uniform, same_figure, sum_to_one, True))
        thread_plot.start()

def attr_cumulative_plot(T,field,by_field='bogus_online',uniform=True,same_figure=False,sum_to_one=True,show_coords=True,x_ticks=15,y_ticks=10):
    """
    print out cumulative distribution of class @by_field as a function of continuous attribute @field
    @T - the DataFrame.
    @same_figure - draw on the figure currently in focus
    @uniform - show @field as distributed uniformly U[0,1]
    @T - table
    """
    # F = T[[field,by_field]][T[field].notnull() & T[by_field].notnull()]
    F = T[[field,by_field]].dropna(how='any')
    F['random_for_sorting'] = np.random.random(len(F))
    F.sort_values(by=[field,'random_for_sorting'],inplace=True)
    del F['random_for_sorting']
    if uniform:
        X = linspace(0,1,len(F))
    else:
        X = array(F[field])

    Y = array(F[by_field])
    cumulative = cumsum(Y)
    if sum_to_one:
        cumulative = cumulative.astype(float)/cumulative[-1]
    else:
        cumulative = cumulative.astype(float)/len(Y)

    if not same_figure:
        fig = plt.figure(random.randint(0,1000))
    plt.plot(X,cumulative,'-')
    stretch_plot()

    if uniform:
        field_displayname = field# +' (uniform)'
    else:
        field_displayname = field

    plt.xlabel(field_displayname)
    plt.ylabel(by_field+' (CDF - Cumulative Distribution Function) ')
    plt.grid(True)
    plt.xticks(linspace(0,1,x_ticks), F[field].iloc[linspace(0,len(F)-1,x_ticks).astype(int)].tolist())
    plt.yticks(linspace(0,1,y_ticks+1), linspace(0,1,y_ticks+1))
    if not same_figure:
        fig.autofmt_xdate()
    plt.draw()
    plt.axes().format_coord = lambda x,y: 'X: %f, Y: %f, %s: %s' % (x, y, field, (str(F[field].iloc[int(x * len(F))]) if x>=0 and int(x * len(F)) < len(F) else '-'))
    #plt.savefig('/tmp/pngs/%s.png' % field_displayname)
    plt.show(block=show_coords) #block=True to display the mouse coords

def plot_betas(roc,**kwargs):
    """ @roc is returned from h2o's .roc() method """
    betas = (array(roc[0]),1-array(roc[1]))
    p = plt.plot(*betas,**kwargs)
    plt.ylabel('beta legit (% of bogus missed)')
    plt.xlabel('beta bogus (% of legit insulted)')
    return p

def attr_scatter_1d(T,field1,by_field='bogus_online',uniform=True,smearY=0.1,smearX=0.01):
    """
    uniform - use only the order of the values, not the actual number, so the distribution is uniform.
    """
    tab = dicts2table( [ {k:rec[k] for k in [field1,by_field]} for rec in T ] )
    # tab = copy.deepcopy(T)
    tab = tab.filter('$%s is not None and $%s is not None'%(field1,by_field))
    tab.sort('$%s + random.random()/10000.'%field1)
    keys = sorted( tab.count(by_field).keys() )

    if np.alltrue([isinstance(k,float) or isinstance(k,int) for k in keys]):
        # numerical values
        val2num = {}
    else:
        # nominal values
        val2num = dict( [(k,i) for i,k in enumerate(keys)] )
    plt.figure( random.randint(0,1000000))
    Y = tab.column('val2num.get($%s,$%s)'%(by_field,by_field))
    if smearY>0:
        R_Y = max(Y) - min(Y) # range of Y values
        Y = tab.column('val2num.get($%s,$%s) + (random.random()-0.5)*R_Y*smearY'%(by_field,by_field))

    if uniform:
        tab.sort(field1)
        X = linspace(0,1,len(tab))
        plt.plot(X,Y,',')
    else:
        R_X = max(tab.column(field1))-min(tab.column(field1)) # range of X values
        X = tab.column('$'+field1 + ' + (random.random()-0.5)*R_X*smearX/100')
        plt.plot(X,Y,',')

    stretch_plot()
    if uniform:
        field1 = field1 +' (uniform)'

    plt.xlabel(field1)
    plt.ylabel(by_field)
    # plt.draw()
    plt.show(block=False)

def attr_scatter(T,field1,field2=None,by_field='bogus_online',uniformX=False,uniformY=False,smearX=0,smearY=0,uniform=None):
    if uniform is not None:
        uniformX = uniformY = uniform
    if field2 is None:
        attr_scatter_1d(T,field1,by_field,uniformX,smearX=smearX)
        return
    # assuming field1,field2 are numerical for a start
    tab = dicts2table( [ {k:rec[k] for k in [field1,field2,by_field]} for rec in T ] )
    tab = tab.filter('$%s is not None and $%s is not None and $%s is not None'%(field1,field2,by_field))
    L = len(tab)

    # adjust fields according to uniformity
    fields = [field1,field2]
    if uniformX or uniformY:
        new_names = [field + '_ind' for field in fields]
        for field,new_field_name,uniform in zip(fields,new_names,[uniformX,uniformY]):
            if not uniform:
                continue
            tab.sort('$%s+random.random()/10000'%field)
            tab.operate(new_field_name,'None')
            for i in range(L):
                tab[i][new_field_name]=i/1./L

    # plot by by_field
    save_tab = tab[:]
    plt.figure( random.randint(0,1000000))
    styles = ['g,','y.','r+','k.'] # ['green','yellow','red','black']
    segments = [(0,0.01-_epsilon), (0.01-_epsilon,0.05+_epsilon), (0.05+_epsilon,1.)]
    for segment in segments: #save_tab.count(by_field):
        tab = save_tab.filter('(%s < $%s < %s)'%(segment[0],by_field,segment[1]))
        # if isinstance(tab[:1].column(field)[0],str):
        #   # discrete field
        #   cnt = tab.count(field)
        #   keys = sorted( cnt.keys() )
        #   D = dict( [(k,i) for i,k in enumerate(keys)] )
        #   X = map(D.get,tab.column(field))

        if uniformX:
            X = tab.column(new_names[0])
        else:
            # determine range to smear
            X = save_tab.column(field1)
            R_X = max(X) - min(X) # range of X values
            # smear
            X = tab.column('$'+field1 + ' + (random.random()-0.5)*R_X*smearX')

        if uniformY:
            Y = tab.column(new_names[1])
        else:
            # determine range to smear
            Y = save_tab.column(field1)
            R_Y = max(Y) - min(Y) # range of Y values
            # smear
            Y = tab.column('$'+field2 + ' + (random.random()-0.5)*R_Y*smearY')

        # if uniform:
        #   plt.plot(tab.column(new_names[0]),tab.column(new_names[1]), styles[0])
        # else:
        #   X = save_tab.column(field1)
        #   R_X = max(X) - min(X) # range of X values
        #   Y = save_tab.column(field2)
        #   R_Y = max(Y) - min(Y) # range of Y values
        #   plt.plot(tab.column('$'+field1 + ' + (random.random()-0.5)*R_X*smearX'),tab.column('$'+field2 + ' + (random.random()-0.5)*R_Y*smearY'),styles[0])

        plt.plot(X,Y,styles[0])
        del styles[0]

    if uniformX:
        field1 = field1 +' (uniform)'
    if uniformY:
        field2 = field2 +' (uniform)'

    plt.xlabel(field1)
    plt.ylabel(field2)
    stretch_plot(1.05)
    # plt.show(block=False)
    # TODO: legend

# example shlifa
ALL_FIELDS = ["eventTimeOnSite", "phonehighHasTwoPhones", "phoneProximityScore", "phonehighNameChain", "namehighClarity", "iphighType", "shipaddhighHHFlag", "distanceShipAddToIp", "iphighProxy", "cardIpBinScoreWith", "billaddhighHHFlag", "distanceBillAddToIp", "emailhighType", "emailhighQuality", "emailhighNameChain", "ipEmailPhoneMatch", "timeDayLight", "commentsLength", "cardAvsResult", "distanceShipAddToBillAdd", "billAmount", "billaddhighType", "billaddhighClarity","blackScore"]

cc_bill_avs = {}
for v in "Y,y,Y|Y,y|y,YYY,YY-,YYX,YYD,YYF,YYM".split(','):
    cc_bill_avs[v]= 4 # full
for v in "a,Y|N,y|n,YNA,YNB,YN-".split(','):
    cc_bill_avs[v]= 3 # ad
for v in "z,w,N|Y,n|y,NYZ,NY-,NYW,NYP".split(','):
    cc_bill_avs[v]= 2 # zip
for v in "n,N|N,n|n,NNN,NN-,NNC".split(','):
    cc_bill_avs[v]= 1 # no
for v in "g,e,s,b,p,r,u,X|X,Z|Z,x|x,z|z,NNI,XX-,XXU,XXR,XXS,XXG".split(','):
    cc_bill_avs[v]= 0 # missing
D = cc_bill_avs
D[''] = D[None] = -1

# T = prepare_table(ALL_FIELDS,limit=100000000)
# F = T.filter('$is_fraud is not None')
# F.operate('cc_bill_avs','D.get($cardAvsResult,-1)')

## auto-discretization

def score_attr_partition(a,b,tab,field,fractional=True):
    """ for instance (100,360,F,'billAmount') - notice you need to maximize """
    tt = tab.filter('$%s is not None and $is_fraud is not None'%field)
    return infor(tt.column('int($%s > %d) + int($%s > %d)'%(field,a,field,b)), tt.column('is_fraud'),fractional=fractional)

# TODO: probabilistic is_fraud (the list @fr)
def score_uniform(tup,fr):
    """
    fr - Table filtered so field is not None, sorted by field, column('is_fraud')
    tup - a list of partitions as percentiles, e.g. [0.3,0.74]
    returns mutual information between discretized field and is_fraud
    """
    L = len(fr)
    # zero out negative values
    stup = [(x+abs(x))/2 for x in sorted(tup)]
    lst = [0]+[int(L*x) for x in stup]+[L]
    cnt = []
    # cnt.append( [fr[:lst[0]].count(False), fr[:tup[0]].count(True)] )
    for i in lrange(lst[:-1]):
        slc = fr[lst[i]:lst[i+1]]
        cnt.append( [slc.count(False),slc.count(True)] )
    # arr = count2arr(cnt)/1./len(T)
    # return cnt
    arr = array(cnt)
    return -infor_probs(probify(arr))

# TODO: program it for 2d and 3d, it's easy.

def find_best_discretization_1d(tab,field,n_buckets=2):
    """
    optimize to find partition with best mutual information between discretized @field and 'is_fraud'
    'is_fraud' is assumed discrete!!
    """
    T = tab.filter('$'+field+' is not None and $is_fraud is not None')
    # sort, randomize order of equal values
    T.sort('$%s + random.random()/10000.'%field)
    fr = T.column('is_fraud')

    x0 = linspace(0,1,n_buckets+1)[1:-1]
    percentiles = scipy.optimize.fmin_powell(score_uniform,x0,args=(fr,))
    if n_buckets == 2:
        percentiles = [percentiles.sum()]
    percentiles = sorted(percentiles)
    explained_frac_entropy = -score_uniform(percentiles,fr) / entropy(fr)
    print 'percentiles: ', percentiles
    print 'I(field ; is_fraud) / H(is_fraud) =',explained_frac_entropy
    L = len(T)
    return [ T.column(field)[int(p*L)] for p in sorted(percentiles) ], explained_frac_entropy

auto_discretize = find_best_discretization_1d

def find_best_discretization_many_buckets(tab,field):
    for i in 2,3,4:
        print i
        print '****'
        print find_best_discretization_1d(tab,field,i)
        print

continuous_fields = "distanceBillAddToIp,distanceShipAddToIp,distanceShipAddToBillAdd,billAmount"

def discretize(tab,continuous_field_name,discrete_field_name,nominal_prefix,partition_values,partition_names=None,  uiter=None):
    """
    in-place.
    @tab - table with @continuous_field_name as a field with numerical values
    @discrete_field_name - name of the discrete field that will be created in tab
    @nominal_prefix - str that will be the prefix of the discrete values. like 'dist'.
    @partition_values - list of the partitions by which to discretize.
    @partition_names - name the different partitions (from smallest to biggest values). None means name by default
    e.g. with [0,30,282] partitions will be (-inf,0], (0,30], (30,282], (282,inf)
    """
    partition_values = sorted(partition_values)
    lst = ['']+map(str,partition_values)+['']
    if partition_names is not None:
        assert len(partition_names) == len(lst)-1, "number of partition_names must be equal to number of partition_values + 1. Received values: %s, names: %s"%(partition_values,partition_names)
        names = partition_names
    else:
        # auto-naming
        names = [nominal_prefix+lst[i]+'_'+lst[i+1] for i in xrange(len(lst[:-1]))]

    if uiter == None:
        uiter = lambda x: x
    tab.operate(discrete_field_name,'None')
    for tx in uiter(tab):
        if tx[continuous_field_name] is None:
            continue
        value = tx[continuous_field_name]
        tx[discrete_field_name] = names[ sum([value>x for x in partition_values]) ]


# langs = merge_tables(txrmd,tt,update_fields='is_fraud')

def tab_inf(field,langs):
    flangs = langs.filter('$is_fraud is not None and $%s is not None'%field)
    return infor(flangs.column(field),flangs.column('is_fraud'))

