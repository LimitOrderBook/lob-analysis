import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../app'))

import frontend.stock_analytics as salib
import numpy as np
import scipy.optimize
from bson import ObjectId
import analysis_lib as al
import json
import numba as nb
import traceback
from numba import jit
import task_lib as tl

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode('ascii')
        if isinstance(obj,scipy.optimize.LbfgsInvHessProduct):
            return '<omitted>'
        return json.JSONEncoder.default(self, obj)
    
#############################################################    
g_x0 = [5.28999, 0.64191979, 1,  0.25]
g_bounds = [(0,100), (0.,0.99), (-3, 10), (0., 10)]    
#############################################################


@jit(nb.float64[:](nb.float64[:], nb.float64[:],nb.float64,nb.float64,nb.float64,nb.float64,nb.float64,nb.float64,nb.float64), nopython=True, nogil=True, cache=True)
def yule_walker_eq( tau, c_tau, c, c_omega, c_beta,phi_dash, g, g_omega, g_beta  ):
    summand = np.zeros(len(tau))

    g_omegak, g_a = al.generate_series_parameters(g_omega, g_beta, K=15, b=5.)
    c_omegak, c_a = al.generate_series_parameters(c_omega, c_beta, K=15, b=5.)

    for i in range(0,len(g_a)):
        for j in range(0,len(c_a)):
            summand += g_a[i]*c_a[j]*g_omegak[i]*c_omegak[j]*\
               (( np.exp(-c_omegak[j]*tau)-np.exp(-g_omegak[i]*tau) ) \
            /(g_omegak[i]-c_omegak[j])  + \
           ( np.exp(-g_omegak[i]*tau) ) /(g_omegak[i]+c_omegak[j]) )


    return  al.c_exp_series(tau, g, g_omegak, g_a)/phi_dash \
    -c_tau \
    + c*g*summand

@jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64,nb.float64,nb.float64,nb.float64,  nb.float64[3]), nopython=True, 
     nogil=True, cache=True)
def min_yule_walker(tau, c_tau,c, c_omega, c_beta, phi_dash, params):
    return np.sum(yule_walker_eq(tau, c_tau, c, c_omega, c_beta, phi_dash, params[0], np.exp(params[1]), np.exp(params[2]))  **2)


def lsq(p, x, y):
    r = al.c_exp_series_wrap(x,p[0],p[1],p[2])
    res = (np.sum((r-y)**2))
    return res
    

def covariance(x,y):
    return ((x-x.mean())*(y-y.mean())).mean()

def calculate_tau_cov_direct(data, dt_range = [0.002,0.004,0.008]):
    res = {}
  
    for dt in dt_range:
        x,y = calculate_tau_cov_direct_dt(data, dt)
        res[dt] = {'tau':x, 'c':y}
    
    
    return res

def calculate_tau_cov_direct_dt(data, dt):
    
    phi_dash = len(data) / data[-1] 
    dtstep = int(dt*1000)
    t = np.arange(0, data[-1], .001)
    N = np.zeros(len(t))
    for i in data:
        N[int(i*1000)] += 1
    N_sum = N.cumsum()
    
    y_tau = []
    x_tau = np.logspace(-2,3,100)
    for tau in x_tau:
        x = (N_sum[dtstep:]-N_sum[:-dtstep])/dt
        y = (N_sum[dtstep + int(tau*1000):]-N_sum[int(tau*1000) :-dtstep])/dt
        x = x[:-len(x)+len(y)]
        y_tau.append(covariance(x,y))

    cov_x = x_tau    
    cov_y = np.array(y_tau)/(phi_dash*phi_dash)
    
    return cov_x, cov_y

def covariance_for_different_dt(data, dt_range = [0.002,0.004,0.008]):
    phi_dash = len(data) / data[-1] 
    res = {}
    for dt in dt_range:
        print(dt)
        salib.toc()
        N = 1000
        tau_array = np.logspace(-2,3,N)
        tau_resarray = np.zeros(N)
        c_resarray = np.zeros(N)
        for i in range(0, N):
            #print(i)
            distance = int(tau_array[i]/dt)
            tau = dt*distance

            t_bins, n_bins, _ = al.dobins(data, stepsize=dt)

            assert ((t_bins[1:]-t_bins[:-1] - dt)**2 < 1e-20).all(), 'must be equally stepped'

            x = n_bins[:-distance]/dt
            y = n_bins[distance:]/dt
            assert len(x) == len(y)
            assert x[distance] == y[0]
            c_resarray[i] = covariance(x,y)/(phi_dash**2)
            tau_resarray[i] = tau
        res[dt] = {'c':c_resarray, 'tau':tau_resarray}
    return res

def fit_moments(data, powerfit=True, lsqfit=False, covar=None, directcovar=True, dt_range = [0.002,0.004,0.008]):
    result = {}
    
    if directcovar:
        result['cov_different_dt'] = calculate_tau_cov_direct(data, dt_range)
    else:
        result['cov_different_dt'] = covariance_for_different_dt(data) if not covar else covar
    result['cov_x'] = np.array([v['tau']  for k,v in result['cov_different_dt'].items()]).mean(axis=0)
    result['cov_y'] = np.array([v['c']  for k,v in result['cov_different_dt'].items()]).mean(axis=0)

    result['cov_fit'] = {}
    if powerfit:
        try:
            print('start powerfit')
            f_power = lambda x,g,omega,beta: g*omega*beta/((1+omega*x)**(1+beta))
            result['cov_fit']['powerfit'] = {'fit_result':scipy.optimize.curve_fit(f_power, result['cov_x'], result['cov_y'], maxfev=int(1e6), bounds=([0.1,1,0.01],[10,1000,10]))}
            c_param = result['cov_fit']['powerfit']['fit_result'][0]
         
            result['cov_fit']['powerfit']['c_param'] = c_param
        except Exception:
            print(traceback.format_exc())
    
    if lsqfit:
        print('start lqfit')
        
        fitresult, fitresult_alt = two_global_fits(lambda params: lsq(params, result['cov_x'], result['cov_y']), ((0.1,10), (1,1000), (0.01,10)), (5, 500, 1))
        
        
        result['cov_fit']['lsqfit']  = {'fit_result':fitresult, 'fit_result_alt':fitresult_alt}
        c_param= result['cov_fit']['lsqfit']['fit_result'].x
        result['cov_fit']['lsqfit']['c_param'] = c_param
        

    # Yule Walker Eq
    x0 = g_x0[1:]
    bounds = [ (0.2,0.9), (-3, 10), (-3,4)]  
    
    phi_dash = len(data) / data[-1] 
    result['g_params'] = {}
    result['fitresult'] = {}
    result['fitresult_alt'] = {}
    for k,v in result['cov_fit'].items():
        
        c, c_omega, c_beta = v['c_param']
        print('start Yule Walker for',k,'using (c,omega,beta)',v['c_param'])
        fityw, fitresult_alt = two_global_fits(lambda params: 
                                min_yule_walker(result['cov_x'], result['cov_y'], c, c_omega, c_beta, phi_dash, params), bounds, x0)
        
        g_param = fityw.x.copy()
        g_param[1] = np.exp(g_param[1])
        g_param[2] = np.exp(g_param[2])
        result['g_params'][k] = g_param
        result['fitresult'][k] = fityw
        result['fitresult_alt'][k] = fitresult_alt
        

    return result

#############################################################

    
    
    
@jit(
    nb.float64(nb.float64[:], nb.float64,nb.float64,nb.float64,nb.float64,nb.float64[:],nb.float64[:],nb.float64[:],nb.float64)
    , nopython=True, nogil=True, cache=True)
def logL_exp_series_news(data, phi_0, g, g_omega, g_beta, 
                    news_t, 
                    news_alpha, 
                    news_beta, 
                    nobookconvention):

    omegak, a = al.generate_series_parameters(g_omega, g_beta, K=15, b=5.)
    
    K = len(a)
    M = len(news_t)
    Z_ik = np.zeros(K)
    
    T = data[-1]
    summand = 0
    for i in range(0, len(data)):
        if i > 0:
            for k in range(0, K):
                Z_ik[k] = (1 + Z_ik[k])*np.exp(-omegak[k]*(data[i] - data[i-1]))
        Z_i = np.sum(Z_ik*a*omegak)
        
        ksummand = 0
        for k in range(0,K):
            ksummand += a[k]*(1- np.exp(-omegak[k]*(T- data[i]))) 
        
        newssummand = 0
        for j in range(0,M):
            if data[i] > news_t[j]:
                newssummand += news_alpha[j]*np.exp(-news_beta[j]*(data[i]-news_t[j]))
        
        summand += np.log(phi_0 + g*Z_i + newssummand) - g*nobookconvention*ksummand
    
    newssummand2 = 0
    for j in range(0,M):
        newssummand2 += news_alpha[j]*(1 - np.exp(-news_beta[j]*(T - news_t[j]) ))/news_beta[j]
        
    return -T*phi_0 + summand - newssummand2

@jit(
    nb.float64(nb.float64[:], nb.float64,nb.float64,nb.float64,nb.float64,nb.float64)
    , nopython=True, nogil=True, cache=True)
def logL_exp_series(data, phi_0, g, g_omega, g_beta, 
                    nobookconvention=1.0): 
    return logL_exp_series_news(data, phi_0, g, g_omega, g_beta, 
                    news_t=np.empty(0, dtype=np.double), 
                    news_alpha=np.empty(0, dtype=np.double), 
                    news_beta=np.empty(0, dtype=np.double), 
                    nobookconvention=nobookconvention)


def two_global_fits(fitfunc, bounds, x0):
    fitresult_2 = scipy.optimize.differential_evolution(fitfunc,maxiter=1000, bounds=bounds )
    if not fitresult_2.success:
        print('differential_evolution WARNING: FIT NOT SUCCESSFUL')
    else:
        print('Fit ok', fitresult_2.x, fitresult_2.fun )
        
    fitresult_1 = scipy.optimize.basinhopping(fitfunc,x0, minimizer_kwargs={
        "bounds":bounds,
         }, niter=200)
    
    if not fitresult_1.lowest_optimization_result.success:
        print('basinhopping WARNING: FIT NOT SUCCESSFUL')
    else:
        print('Fit ok', fitresult_1.x, fitresult_1.fun)
        
    
        
    diff =  fitresult_1.fun - fitresult_2.fun
    print('values for basinhopping, differential_evolution, difference',fitresult_1.fun , fitresult_2.fun, diff)
    if diff > 0:
        fitresult = fitresult_2
        fitresult_alt = fitresult_1
    else:
        fitresult = fitresult_1
        fitresult_alt = fitresult_2
    
    return fitresult, fitresult_alt 


def fit_mle_single_news_term(data, phi_0, g, g_omega, g_beta, x0=(1,2,3), bounds=[(0, 1e10), (0, 1e10), (-3,10)]  ):
     
    fitfunc = lambda params: -logL_exp_series_news(data,  phi_0, g, g_omega, g_beta,  
                                              np.array([params[0]]),
                                              np.array([params[1]]),
                                              np.array([np.exp(params[2])]),
                                              nobookconvention=1.)

    fitresult, fitresult_alt = two_global_fits(fitfunc, bounds, x0)
        
    news_params = fitresult.x.copy()
    news_params[2] = np.exp(news_params[2])
    return {'news_params':news_params, 'fitresult':fitresult, 'alternative_result':fitresult_alt}



def fit_mle(data, book_convention=False, fix_phi_dash=True):
    # FIT USING BOOK MLE W/O FIXED PHI
    x0 = g_x0
    bounds = g_bounds
    
    fitfunc = None
    if fix_phi_dash:
        phi_dash = len(data) / data[-1]
        fitfunc = lambda params: -logL_exp_series(data, phi_dash * (1-params[0]), params[0],np.exp(params[1]),params[2],  
                                                  nobookconvention=-1. if book_convention else 1.)
        x0 = x0[1:]
        bounds = bounds[1:]
    else:
        fitfunc = lambda params: -logL_exp_series(data, params[0],params[1],np.exp(params[2]),params[3],
                                                  nobookconvention=-1. if book_convention else 1.)

    fitresult, fitresult_alt = two_global_fits(fitfunc, bounds, x0)
        
    g_params = fitresult.x if fix_phi_dash else fitresult.x[1:]
    g_params[1] = np.exp(g_params[1])
    return {'g_params':g_params, 'fitresult':fitresult, 'alternative_result':fitresult_alt} if fix_phi_dash else \
                {'g_params':g_params, 'phi_0':fitresult.x[0], 'alternative_result':fitresult_alt,
                 'fitresult':fitresult}

def fit_mle_extended(data, t_0, dtbound = 120):
    # Fit g_params on first part
    first_part = data[data < t_0]
    first_part_res = fit_mle(first_part)
    
    g, g_omega, g_beta = first_part_res['g_params']

    phi_dash = len(first_part) / first_part[-1]
    phi_0 = phi_dash * (1-g)

    print('phi_0, g, g_omega, g_beta',phi_0, g, g_omega, g_beta  )
    
    # Fit both parts using extended mle
    
    both_part_res = fit_mle_single_news_term(data, phi_0, g, g_omega, g_beta,
                                  x0=(1,2,3), # t, alpha, beta
                                   bounds=[(t_0 - dtbound, t_0 + dtbound), (0, 1e2), (-10, 5)])
    
    both_part_res['first_part'] = first_part_res
    both_part_res['first_part']['phi_dash'] = phi_dash
    return both_part_res
    
#################################################
def do_covariance(param):
    data = load_data(param)
    return covariance_for_different_dt(data)

def do_simulate(param):
    data = None
    N = 0
    if 'N' in param:
        N = param['N']
    else:
        data = load_data(param)
        N = len(data)
    pd = ( len(data) / data[-1]  if 'phi_dash' not in param else param['phi_dash'] )  \
        if 'phi_0' not in param else param['phi_0']/(1-param['g_params'][0])
    print('simulate using phi_dash', pd)
    print('simlength',N)
    
    news_params= np.array([[]],dtype=np.double) if 'news_params' not in param else np.array(param['news_params'])
    by_itrans = 'by_itrans' in param and param['by_itrans']
    if by_itrans:
        assert 'news_params' not in param, 'not supported'
        sim_results = al.simulate_by_itrans(phi_dash=pd,
                                         g_params=tuple(param['g_params']),  K=15, N=N)
    else:
        
        sim_results = al.simulate_by_thinning(phi_dash=pd,
                                         g_params=tuple(param['g_params']),  K=15, news_params=news_params, N=N, caching=False)
    
    return {'sim_results':sim_results}

def load_data(param, randomize=True):
    ticks = salib.stock_analytics(param['id'], gui_mode=False).ticks.aggregate([{"$match":{"timestamp":{"$gte":param['start'], "$lt":param['stop']}
                              ,"type":{"$in":["fill","execute","trade"]}}
                              },{"$project":{"timestamp":1}}])
    
    data = np.array([a['timestamp'] for a in ticks])
    if randomize:
        data += np.random.rand(len(data))*1.
    data.sort()
    data = (data-data[0])/1000
    return data
def load_cached_covar(param):
    tbl = tl.dbconnect()
    res = list(tbl.aggregate([{"$match":{"status":3,"error":None,"task.task":"covariance",
                                         "task.start":param['start'],
                                          "task.stop":param['stop'],
                                         "task.id":param['id']
                                         }},{"$sort":{"task.id":1}}]))
    assert len(res) == 1, res
    return res[0]['result']

def load_sim_data(oid, discard_first=0):
    if not isinstance(oid, ObjectId):
        oid = ObjectId(oid)
    
    tbl = tl.dbconnect()
    res = list(tbl.aggregate([{"$match":{"status":3,"error":None,"_id":oid}}]))
    assert len(res) == 1
    
    data = np.array(res[0]['result']['sim_results'])
    data.sort()
    
    if discard_first > 0:
        print('discard_first',discard_first)
        
    data = data[discard_first:]
    
    data = data-data[0]
    return data

def do_fit(param): 
    data = load_sim_data(param['load_sim_data_w_id' ], discard_first=(param['discard_first'] if 'discard_first' in param else 0) ) if 'load_sim_data_w_id' in param else load_data(param)
    
    if param['method'] == 'mle':
        return fit_mle(data, param['book_convention'], param['fix_phi_dash'])
    
    elif param['method'] == 'moments':
        covar = None
        if param['loadcachedcovariance']:
            covar = load_cached_covar(param)
        return fit_moments(data, param['powerfit'], param['lsqfit'], covar=covar, directcovar=param['directcovar'], dt_range=param['dt_range'])
    
    elif param['method'] == 'mle_extended':
        return fit_mle_extended(data, param['t_0'], dtbound=param['dtbound'])

    assert False,'Method not supported'