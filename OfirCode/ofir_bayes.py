import sys
# if './art' not in sys.path:
#     sys.path.append('./art/')
from art import *
# if './art/regev' not in sys.path:
#     sys.path.append('./art/regev')
# from utils import * # count,...
from numpy import *

_epsilon = 0.0000000000000000001 # avoid 0 division and log

def fractional_count(X,Y=None,Z=None):
    """
    performs a 1/2/3-dim count with fractional balls.
    @X,[,@Y,@Z] - aligned lists of discrete values. last variable is the probability of a binary variable to be True
    None values are ignored
    """
    cnt = {}
    if Y is None:
        # count X
        fX = filter(lambda x: x is not None, X)
        s = sum(fX)
        cnt = {True:s,False:len(fX)-s}
        cnt.pop(None,None)
        return cnt
    elif Z is None:
        # count X,Y
        for x,y in zip(X,Y):
            cnt[(x,True)]  = cnt.get((x,True) ,0) +   y
            cnt[(x,False)] = cnt.get((x,False),0) + 1-y
    else:
        # count X,Y,Z
        for x,y,z in zip(X,Y,Z):
            cnt[(x,y,True)]  = cnt.get((x,y,True) ,0) +   z
            cnt[(x,y,False)] = cnt.get((x,y,False),0) + 1-z

    # ignore None values
    for k in cnt.keys():
        if None in k:
            cnt.pop(k)
    return cnt

def infor_probs(pr_X_Y):
    """ @pr_X_Y - 2 dim array of joint probability p(x,y). returns I(X;Y) """
    pr_X = sum(pr_X_Y,1)
    return entropy_probs(pr_X) - conditional_entropy_count(pr_X_Y)

frac_infor = lambda X,Y,Z=None: infor(X,Y,Z,True)

def infor(X,Y,Z=None,fractional=False):
    """
    Mutual information of X,Y (given Z): I(X;Y|Z)
    @fractional = True - Y/Z is the probability of being True, not the value
    """

    #dist = joint_distribution(X,Y,Z,fractional=fractional)

    # unconditional
    if Z is None:
        # is this needed? we filter out later.
        # temp_X = [X[i] for i in range(len(X)) if X[i] is not None and Y[i] is not None]
        # temp_Y = [Y[i] for i in range(len(X)) if X[i] is not None and Y[i] is not None]

        # max to avoid truncation errors
        # return max( 0, entropy(temp_X) - conditional_entropy(temp_X,temp_Y,fractional=fractional) )
        return max( 0, entropy(X) - conditional_entropy(X,Y,fractional=fractional) )


    # temp_X = [X[i] for i in range(len(X)) if X[i] is not None and Y[i] is not None and Z[i] is not None]
    # temp_Y = [Y[i] for i in range(len(X)) if X[i] is not None and Y[i] is not None and Z[i] is not None]
    # temp_Z = [Z[i] for i in range(len(X)) if X[i] is not None and Y[i] is not None and Z[i] is not None]

    # conditional
    # ===========
    # joint count
    # cnt_X_Y_Z = count2arr(joint_distribution(temp_X,temp_Y,temp_Z,fractional=fractional))
    cnt_X_Y_Z = count2arr(joint_distribution(X,Y,Z,fractional=fractional))
    cnt_Z = sum(cnt_X_Y_Z,(0,1))
    # probability(Z)
    pr_Z = probify(cnt_Z)
    pr_X_Y_given_Z = cnt_X_Y_Z /1./ (cnt_Z + _epsilon) # + epsilon to avoid 0/0

    # I(X;Y|Z=z) for each z
    # AAARGH!! This can be done in numpy, but I'm lazy. TODO.
    lst = []
    for z in xrange(cnt_X_Y_Z.shape[2]):
        lst.append( infor_probs(pr_X_Y_given_Z[:,:,z]) )

    # Ez[I(X;Y|Z=z)]
    return max( 0, sum( array(lst) * pr_Z ) )


def count2arr(cnt):
    """ returns N-dim array with count """
    keys = cnt.keys()
    n_dims = len(keys[0])
    dims = []
    val_dicts = []
    for d in xrange(n_dims):
        vals = sorted(list(set(col(keys,d))))
        D = dict( [(v,i) for i,v in enumerate(vals)] )
        val_dicts.append(D)
        dims.append(len(vals))
        #
    arr = zeros(dims)
    for k in cnt:
        # convert raw value to indexed value for each variable
        index = tuple([val_dicts[i][k[i]] for i in xrange(n_dims)])
        # print index, k, cnt[k]
        arr.__setitem__(index,cnt[k])
        #
    return arr



def joint_distribution(X,Y,Z=None,fractional=False):
    """
    X,Y[,Z] are aligned lists of discrete values
    @fractional = True - Y/Z is the probability of being True, not the value
    None values are ignored
    """
    if not fractional:
        if Z is None:
            cnt = count( zip(X,Y) )
        else:
            cnt = count( zip(X,Y,Z) )

        # ignore None values
        for k in cnt.keys():
            if None in k:
                cnt.pop(k)
        return cnt
    else:
        return fractional_count(X,Y,Z)

def entropy_count(count_arr):
    """ count_arr may be multidimensional """
    return entropy_probs(probify(count_arr))

def entropy_probs(probs):
    """ calculates entropy from probability N-dim array """
    arr = array(probs)
    return -sum( log2(arr+_epsilon)*arr ) # + epsilon to avoid log(0)

def conditional_entropy(X,Y,fractional=False):
    """ H(X|Y) """
    return conditional_entropy_count(joint_distribution(X,Y,fractional=fractional))

def conditional_entropy_count(cnt):
    """ joint count (or probs) of X,Y -> H(X|Y) """
    if isinstance(cnt,dict):
        cnt_X_Y = count2arr(cnt)
    else:
        cnt_X_Y = cnt.copy()

    cnt_Y = sum(cnt_X_Y,0)
    probs_Y = probify(cnt_Y)
    probs_X_given_Y = cnt_X_Y / (cnt_Y + _epsilon) # + epsilon to avoid 0./0.
    # H(X|Y=y)
    H_X_given_y = -sum( log2(probs_X_given_Y+_epsilon)*probs_X_given_Y, 0) # + epsilon to avoid log(0)
    # H(X|Y)
    H_X_given_Y = sum( probs_Y * H_X_given_y )

    return H_X_given_Y

def entropy(X,Y=None,fractional=False):
    """
    observed entropy of X (and Y): H(X) OR H(X,Y)
    @fractional = True - X/Y is the probability of being True, not the value
    """
    if Y is not None:
        cnt = joint_distribution(X,Y,fractional=fractional)
    elif fractional:
        cnt = fractional_count(X)
    else:
        cnt = count(X)
        cnt.pop(None,None)

    probs = probify(cnt.values())
    return entropy_probs(probs)
