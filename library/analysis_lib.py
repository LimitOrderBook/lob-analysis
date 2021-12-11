import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize
import frontend.stock_analytics as salib
import numba as nb
from numba import jit

@jit(nb.types.UniTuple(nb.float64[:],2)(nb.float64,nb.float64,nb.int32,nb.float64), nopython=True, nogil=True, cache=True)
def generate_series_parameters(g_omega, g_beta, K=15, b=5.):
    k = np.arange(0,K,1) 
    omegak = g_omega/(b**k)
    a = omegak**g_beta
    a /= np.sum(a)
    return omegak, a
    
def c_exp_series_wrap(tau, g, g_omega, g_beta, K=15, b=5.):
    omegak, a = generate_series_parameters(g_omega, g_beta, K, b)
    return c_exp_series(tau, g, omegak, a)

@jit(nb.float64[:](nb.float64[:], nb.float64, nb.float64[:],nb.float64[:]), nopython=True, nogil=True, cache=True)
def c_exp_series(tau, c, omegak, a):
    return c*np.sum(np.multiply ( np.multiply (  np.exp(-np.outer(omegak ,tau)) .T, omegak), a ), axis=1)


def dobins(ts_array, N = 1000, x_bins=None, useinteger=False, stepsize=None,  ignoreabove=False):
    ts_array.sort()
   
    if x_bins is None:
        if useinteger:
            minp = math.floor(ts_array[0])
            maxp = math.ceil(ts_array[-1])
            steps = stepsize if stepsize is not None else np.ceil((maxp-minp)/N)

            x_bins = np.arange(minp, maxp+2, steps)


            
        else:
            if stepsize is None:
                stepsize = (ts_array[-1]-ts_array[0])/N
            x_bins = np.arange(ts_array[0], ts_array[-1]+2*stepsize, stepsize)
            
    stepsize = x_bins[1]-x_bins[0]
    N = len(x_bins)-1
    
    
    dt = x_bins[1]-x_bins[0]

    y_bins = np.zeros(len(x_bins))
    
    unique, counts = np.unique(np.floor((ts_array-x_bins[0])/dt), return_counts=True)
    
    if ignoreabove:
        for a,b, in zip(unique.astype(int), counts):
            if a < len(y_bins):
                y_bins[a] = b
    else:
        y_bins[unique.astype(int)] = counts#[:-1]
    
    while not ignoreabove and x_bins[-1] >= ts_array[-1]:
        x_bins = x_bins[:-1]
        y_bins = y_bins[:-1]

    x_bins += stepsize/2.
    
    E  = y_bins.mean()
    V = y_bins.var()

    return x_bins, y_bins, V/E


def print_stats(ats_array, tau = np.logspace(-1,3,20), N=1000, splitpoint=None,stepsize_hist=2.):
    
    if len(ats_array) > 20:
        ats_array = [ats_array]
        
    plt.rcParams['figure.figsize'] = (15, 15)
    grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0.3)
    
    for kts_array in ats_array:
        if type(kts_array) is tuple:
            ts_array = kts_array[1]
            label = kts_array[0]
        else:
            ts_array = kts_array
        
        plt.subplot(grid[0, 0:2])
        x_bins, y_bins, _ = dobins(ts_array, N = N)
        plt.plot(x_bins, y_bins, label=label)
        plt.legend()
        plt.subplot(grid[0, 2])

        if splitpoint is not None:
            y_bins1 = y_bins[:int(splitpoint*len(y_bins))]
            y_bins2 = y_bins[int(splitpoint*len(y_bins)):]
            a_bins1, b_bins1, _ = dobins(y_bins1, useinteger=True, N = 25)
            a_bins2, b_bins2, _ = dobins(y_bins2, useinteger=True, N = 25)
            plt.plot(b_bins1, a_bins1, label=label)
            plt.plot(b_bins2, a_bins2, label=label)
            print('(1) V =',y_bins1.var(),'; E =',y_bins1.mean(),'; V/E =', y_bins1.var()/y_bins1.mean())
            print('(2) V =',y_bins2.var(),'; E =',y_bins2.mean(),'; V/E =', y_bins2.var()/y_bins2.mean())

        a_bins, b_bins, _ = dobins(y_bins, useinteger=True, stepsize=stepsize_hist)
        plt.plot(b_bins, a_bins, label=label)
        
       
        print('V =',y_bins.var(),'; E =',y_bins.mean(),'; V/E =', y_bins.var()/y_bins.mean())
        plt.subplot(grid[1, :])


        r = calc_r_tau(ts_array, tau)

        f = lambda tau,beta,A: A/(tau**beta)
        fitted = scipy.optimize.curve_fit(f, tau,np.sqrt(1/r))

        plt.loglog(tau,np.sqrt(1/r) , label=label)
        plt.loglog(tau,f(tau, fitted[0][0], fitted[0][1]), label=label+' fitted' )
        plt.legend()
        
        plt.subplot(grid[2, :])
        plt.plot(tau,r , label=label)
        plt.legend()
      
    plt.show()
    plt.rcParams['figure.figsize'] = (15, 5)


def calc_r_tau(ts_array, tau):
    r = np.zeros(len(tau))
    for i in range(0,len(tau)):
        _,_,rr = dobins(ts_array, stepsize=tau[i])
        r[i] = rr
    return r


g_cache_dict = {}


@jit(nb.float64(nb.float64, nb.float64[:], nb.int64, nb.float64[:],nb.float64[:]), nopython=True, nogil=True, cache=True)
def c_exp_series_sum(t, tau, uptoi, omegak, a):
    return np.sum(np.multiply ( np.multiply (  np.exp(-np.outer(omegak ,t-tau[:uptoi])) .T, omegak), a ))

@jit(nb.float64[:](nb.float64, nb.types.UniTuple(nb.float64,3),nb.int64,nb.float64[:,:],nb.int64,nb.boolean,nb.boolean), nopython=True, nogil=True, cache=True)
def simulate_by_thinning_nocache(phi_dash, g_params, K, news_params, N = 250000, reseed=True, status_update=True):
    # Initialize parameters
    g, g_omega, g_beta = g_params
    phi_0 = phi_dash * (1-g)
    
    omegak, a = generate_series_parameters(g_omega, g_beta, K, b=5.)


    if reseed:
        np.random.seed(124)
    #salib.tic()
    i = randi1i = randi2i = 0
    t = 0.
    randpool1 = - np.log(np.random.rand(100*N))
    randpool2 = np.random.rand(100*N)


    # Thinning algorithm
    hawkes_array = np.zeros(N)
    i = 0
    
    while i < N: 
        lambda_star = phi_0 + g*c_exp_series_sum(t,hawkes_array,i, omegak, a)  
        
        for nn in range(0, len(news_params)):
            n = news_params[nn]
            if t > n[0]:
                lambda_star += n[1]*np.exp(-n[2]*(t-n[0]))
        
        if lambda_star < 1e-100:
            print('warnung: abbruch')
            return hawkes_array[:i]
        
        u = randpool1[randi1i]
        randi1i+=1
        tau = u / lambda_star

        t += tau
        
        s = randpool2[randi2i]
        randi2i+=1
        
        
        lambda_star2 = phi_0 + g*c_exp_series_sum(t,hawkes_array,i, omegak, a) 
        for nn in range(0, len(news_params)):
            n = news_params[nn]
            if t > n[0]:
                lambda_star2 += n[1]*np.exp(-n[2]*(t-n[0]))
        
        if s <=  lambda_star2 / lambda_star:
            hawkes_array[i] = t
            i += 1
            if status_update and i%(int(N/5))==0:
                print(i)

    return hawkes_array


#################################
# Theoretische Verteilung
def binary_search( f, target, cstep=10, stepsize=10, prevturn=True): # mon increasing func
    if cstep > 1e5:
        return -1
    res = target/f(cstep) 
    if np.abs(res-1) < 1e-4:
        return cstep
    
    if res < 1:  
        stepsize /= 2
        prevturn=False
        cstep -= stepsize 
    else:
        if prevturn:
            stepsize *= 2
        else:
            stepsize /= 2
        cstep += stepsize 
    return binary_search( f, target, cstep, stepsize,prevturn)

def integral_over_phi_slow(t,deltat, omegak, a, K, phi_0,g):
    summand = 0
    
    if len(t) > 0:
        for k in range(0,K):
            summand += (1-np.exp(-omegak[k]*deltat))*np.sum(a[k]*np.exp(-omegak[k]*(t[-1]-t)))

    return deltat*phi_0 + g*summand

def integral_over_phi(t,deltat, omegak, a, K, phi_0,g):
   
    summand = np.sum((1-np.exp(-np.outer(omegak,deltat))).T * np.sum(np.multiply(np.exp(-np.outer(omegak,(t[-1]-t))).T,a), axis=0)  ,axis=1) \
        if len(t) > 0 else 0
    return deltat*phi_0 + g*summand
            
def probability_for_inter_arrival_time(t, deltat, omegak, a, K, phi_0,g):

    x= integral_over_phi(t,deltat, omegak, a, K, phi_0,g)
    return 1-np.exp(-x)
def probability_for_inter_arrival_time_slow(t, deltat, omegak, a, K, phi_0,g):
    x = np.zeros(len(deltat))
    for i in range(0, len(deltat)):
        x[i]= integral_over_phi_slow(t,deltat[i], omegak, a, K, phi_0,g)
    return 1-np.exp(-x)

def simulate_by_itrans(phi_dash, g_params, K, conv1=1e-8, conv2=1e-2, N = 250000, init_array=np.array([]), reseed=True, status_update=True, use_binary_search=True):
    print('simulate_by_itrans')
    # Initialize parameters
    g, g_omega, g_beta = g_params
    
    phi_0 = phi_dash * (1-g)
    
    omegak, a = generate_series_parameters(g_omega, g_beta, K, b=5.)
    

    if reseed:
        np.random.seed(123)
    salib.tic()
    i = randii = 0
    t = 0.
    randpool = np.random.rand(100*N)


    # Inverse transform algorithm
    init_array = np.array(init_array, dtype='double')
    hawkes_array = np.pad(init_array,(0,N-len(init_array)), 'constant', constant_values=0.)   #np.zeros(N)
    hawkes_array = np.array(hawkes_array, dtype='double')
    i = len(init_array)
    if i > 0:
        t = init_array[-1]
        
    endsize = 20
    tau = 0
    while i < N: 
        NN = 10000
        
        u = randpool[randii]
        randii+=1
        if randii >= len(randpool):
            print(i)
        
        
        if use_binary_search:
            f = lambda x: probability_for_inter_arrival_time(hawkes_array[:i],x, omegak, a, K, phi_0, g)
            tau = binary_search( f, u,cstep=max(tau,1e-5), stepsize=max(tau,1e-5))
            if tau == -1:
                return hawkes_array[:i]
        else:
            notok = 1
            while notok>0:
                if notok > 10:
                    NN *= 2
                    notok = 1
                tau_x = np.linspace(0,endsize,NN)
                pt = probability_for_inter_arrival_time      (hawkes_array[:i],tau_x, omegak, a, K, phi_0, g)

                okok = True
                if pt[-1]-pt[-2] > conv1:
                    if status_update:
                        print('warning, pt does not converge',i,pt[1]-pt[0],pt[-1]-pt[-2])
                    endsize*=1.1
                    notok += 1
                    okok = False
                if pt[1]-pt[0] > conv2:
                    if status_update:
                        print('warning pt increases to fast',i,pt[1]-pt[0],pt[-1]-pt[-2])
                    endsize/=1.1
                    notok +=1
                    okok = False
                if okok:
                    notok = 0
                    
            tt = np.max(np.where(pt < u))
            if tt == NN-1:
                if status_update:
                    print('vorzeitig abgebrochen', u, tau_x[tt], pt[tt])
                return hawkes_array[:i]
            tau = tau_x[tt] 
        
        t += tau
        
        hawkes_array[i] = t
        i += 1
        if status_update and i%(int(N/5))==0:
            print(i)
            salib.toc()
    if status_update:
        salib.toc()        
    return hawkes_array
#############################

def simulate_by_thinning(phi_dash, g_params, K, news_params=np.array([[]],dtype=np.double), N = 250000, caching=False, init_array=np.array([]), reseed=True, status_update=True):
    if not caching:
        assert init_array is not np.array([]), 'unsupported'
        return simulate_by_thinning_nocache(phi_dash, g_params, K, news_params, N, reseed, status_update)
    
    print('warning: using cached version')
    # Initialize parameters
    g, g_omega, g_beta = g_params
    phi_0 = phi_dash * (1-g)
    
    omegak, a = generate_series_parameters(g_omega, g_beta, K)

    
    
    hawkes_process = None
    
    
    if caching:
        assert len(news_params) == 0, 'not supported'
        hawkes_process = lambda t: phi_0 + np.sum(cache_dict[(t*accur).astype(int)])
    
        lowinfluence_time = 3/g_omega

        accur = 10000 # 0.1ms
        NN = int(np.ceil(accur* lowinfluence_time))
        
        ckey = json.dumps({'a':NN, 'b':accur, 'c':g,'d':list(omegak), 'e':list(a)})
        if ckey not in g_cache_dict:
            cache_dict = np.zeros(NN)
            for i in range(0, NN):
                cache_dict[i] = c_exp_series(i/accur, g,omegak, a)
            g_cache_dict[ckey] = cache_dict
        cache_dict = g_cache_dict[ckey]
    else:
        assert False, 'please use numba version'
        hawkes_process = lambda current_t, previous_t: phi_0 + np.sum(c_exp_series(current_t-previous_t, g,omegak, a)) + \
            np.sum(np.array((news_params[:,0]<current_t), dtype=np.double)*news_params[:,1]*np.exp(-news_params[:,2]*(current_t-news_params[:,0])))

    if reseed:
        np.random.seed(124)
    salib.tic()
    i = j = randii = 0
    t = 0.
    randpool = np.random.rand(100*N)


    # Thinning algorithm
    init_array = np.array(init_array, dtype='double')
    hawkes_array = np.pad(init_array,(0,N-len(init_array)), 'constant', constant_values=0.)   #np.zeros(N)
    hawkes_array = np.array(hawkes_array, dtype='double')
    i = len(init_array)
    if i > 0:
        t = init_array[-1]
        
    while i < N: 
        lambda_star = hawkes_process(t,hawkes_array[j:i] )

        
        if lambda_star < 1e-100:
            return hawkes_array[:i]
        
        if randii >= len(randpool):
            print(i)
        u = randpool[randii]
        randii+=1
        tau = - np.log(u) / lambda_star

        t += tau
        
        while caching and hawkes_array[j] <= t-lowinfluence_time and j < i:
            j+=1
                
        if randii >= len(randpool):
            print(i)
        s = randpool[randii]
        randii+=1
        if s <= hawkes_process(t- hawkes_array[j:i]) / lambda_star:
            hawkes_array[i] = t
            i += 1
            if status_update and i%(int(N/5))==0:
                print(i, j)
                salib.toc()
    if status_update:
        salib.toc()        
    return hawkes_array