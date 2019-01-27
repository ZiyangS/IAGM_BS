import sys
import numpy as np
from scipy.stats import gamma, invgamma, wishart, norm
from scipy.stats import multivariate_normal as mv_norm
from scipy import special
from ars import ARS
import mpmath
from numba import jit, njit, autojit, vectorize, guvectorize, float64, float32
from functools import lru_cache

# the maximum positive integer for use in setting the ARS seed
maxsize = sys.maxsize


def compare_s_ljk(s_ljk, previous_s_ljk, s_rjk, nj, beta, w, sum):
    s_ljk = mpmath.mpf(s_ljk)
    s_rjk = mpmath.mpf(s_rjk)
    a1 = mpmath.power(s_ljk, -0.5) + mpmath.power(s_rjk, -0.5)
    a2 = mpmath.power(previous_s_ljk, -0.5) + mpmath.power(s_rjk, -0.5)
    ratio_a = a2/a1
    ratio_a_power = np.power(ratio_a, nj)
    ratio_b = mpmath.power(s_ljk, (beta/2-1)) * mpmath.exp(-0.5*s_ljk*sum) * mpmath.exp(-0.5*w*beta*s_ljk) \
            / (mpmath.power(previous_s_ljk, (beta/2-1)) * mpmath.exp(-0.5*previous_s_ljk*sum) * mpmath.exp(-0.5*w*beta*previous_s_ljk))
    # print(ratio_a_power * ratio_b)
    return ratio_a_power * ratio_b


def Metropolis_Hastings_Sampling_posterior_sljk(s_ljk, s_rjk, nj, beta, w, sum):
    n = 500
    x = s_ljk
    vec = []
    vec.append(x)
    for i in range(n):
        # proposed distribution make sure 25%-40% accept
        # random_walk algorithm, using symmetric Gaussian distribution, so it's simplified to Metropolis algoritm
        # the parameter is mu: the previous state of x and variation
        candidate = norm.rvs(x, 0.75, 1)[0]
        if candidate <= 0:
            candidate = np.abs(candidate)
        # acceptance probability
        alpha = min([1., compare_s_ljk(candidate, x, s_rjk, nj, beta, w, sum)])
        # alpha = min([1., posterior_distribution_s_ljk(candidate, s_rjk, nj, beta, w, sum)/
        #              posterior_distribution_s_ljk(x, s_rjk, nj, beta, w, sum)])
        u = np.random.uniform(0,1)
        if u < alpha:
            x = candidate
            vec.append(x)
    return vec[-1]


def compare_s_rjk(s_rjk, previous_s_rjk, s_ljk, nj, beta, w, sum):
    s_ljk = mpmath.mpf(s_ljk)
    s_rjk = mpmath.mpf(s_rjk)
    a1 = mpmath.power(s_ljk, -0.5) + mpmath.power(s_rjk, -0.5)
    a2 = mpmath.power(s_ljk, -0.5) + mpmath.power(previous_s_rjk, -0.5)
    ratio_a = a2/a1
    ratio_a_power = np.power(ratio_a, nj)
    ratio_b = mpmath.power(s_rjk, (beta/2-1)) * mpmath.exp(-0.5*s_rjk*sum) * mpmath.exp(-0.5*w*beta*s_rjk) \
            / (mpmath.power(previous_s_rjk, (beta/2-1)) * mpmath.exp(-0.5*previous_s_rjk*sum) * mpmath.exp(-0.5*w*beta*previous_s_rjk))
    # print(ratio_a_power * ratio_b)
    return ratio_a_power * ratio_b


def Metropolis_Hastings_Sampling_posterior_srjk(s_ljk, s_rjk, nj, beta, w, sum):
    n = 500
    x = s_rjk
    vec = []
    vec.append(x)
    for i in range(n):
        # proposed distribution make sure 25%-40% accept
        # random_walk algorithm, using symmetric Gaussian distribution, so it's simplified to Metropolis algoritm
        # the parameter is mu: the previous state of x and variation
        candidate = norm.rvs(x, 0.75, 1)[0]
        if candidate <= 0:
            continue
        # acceptance probability
        alpha = min([1., compare_s_rjk(candidate, x, s_ljk, nj, beta, w, sum)])
        # alpha = min([1., posterior_distribution_s_rjk(candidate, s_ljk, nj, beta, w, sum)/
        #              posterior_distribution_s_rjk(x, s_ljk, nj, beta, w, sum)])
        u = np.random.uniform(0,1)
        if u < alpha:
            x = candidate
            vec.append(x)
    return vec[-1]


@jit(nogil=True,)
def Asymmetric_Gassian_Distribution_pdf(x_k, mu_jk, s_ljk, s_rjk):
    y_k = np.zeros(x_k.shape[0])
    for i, xik in enumerate(x_k):
        if xik < mu_jk:
            y_k[i] = np.sqrt(2/np.pi)/(np.power(s_ljk, -0.5) + np.power(s_rjk, -0.5))\
                   * np.exp(- 0.5 * s_ljk * (xik- mu_jk)**2)
        else:
            y_k[i] = np.sqrt(2/np.pi)/(np.power(s_ljk, -0.5) + np.power(s_rjk, -0.5))\
                   * np.exp(- 0.5 * s_rjk * (xik- mu_jk)**2)
    return y_k


def integral_approx(X, lam, r, beta_l, beta_r, w_l, w_r, size=25):
    """
    estimates the integral, eq 17 (Rasmussen 2000)
    """
    N, D = X.shape
    temp = np.zeros(len(X))
    i = 0
    while i < size:
        # mu = np.array([np.squeeze(norm.rvs(loc=lam[k], scale=1/r[k], size=1)) for k in range(D)])
        mu = draw_MVNormal(mean=lam, cov=1/r)
        s_l = np.array([np.squeeze(draw_gamma(beta_l[k] / 2, 2 / (beta_l[k] * w_l[k]))) for k in range(D)])
        s_r = np.array([np.squeeze(draw_gamma(beta_r[k] / 2, 2 / (beta_r[k] * w_r[k]))) for k in range(D)])
        ini = np.ones(len(X))
        for k in range(D):
            # use metropolis-hastings algorithm to draw sampling from AGD
            # the size parameter is the required sampling number which is equal to the dataset's number
            # the n parameter is MH algorithm itering times,because the acceptance rate should be 25%-40%
            temp_para = Asymmetric_Gassian_Distribution_pdf(X[:, k], mu[k], s_l[k], s_r[k])
            ini *= temp_para
        temp += ini
        i += 1
    return temp/float(size)


def log_p_alpha(alpha, k, N):
    """
    the log of eq15 (Rasmussen 2000)
    """
    return (k - 1.5)*np.log(alpha) - 0.5/alpha + special.gammaln(alpha) - special.gammaln(N + alpha)


def log_p_alpha_prime(alpha, k, N):
    """
    the derivative (wrt alpha) of the log of eq 15 (Rasmussen 2000)
    """
    return (k - 1.5)/alpha + 0.5/(alpha*alpha) + special.psi(alpha) - special.psi(alpha + N)


def log_p_beta(beta, M, cumculative_sum_equation=1):
    return -M*special.gammaln(beta/2) \
        - 0.5/beta \
        + 0.5*(beta*M-3)*np.log(beta/2) \
        + 0.5*beta*cumculative_sum_equation

def log_p_beta_prime(beta, M, cumculative_sum_equation=1):
    return -M*special.psi(0.5*beta) \
        + 0.5/beta**2 \
        + 0.5*M*np.log(0.5*beta) \
        + (M*beta -3)/beta \
        + 0.5*cumculative_sum_equation

def draw_beta_ars(w, s, M, k, size=1):
    D = 2
    cumculative_sum_equation = 0
    for sj in s:
        cumculative_sum_equation += np.log(sj[k])
        cumculative_sum_equation += np.log(w[k])
        cumculative_sum_equation -= w[k]*sj[k]
    lb = D
    ars = ARS(log_p_beta, log_p_beta_prime, xi=[lb + 15], lb=lb, ub=float("inf"), \
             M=M, cumculative_sum_equation=cumculative_sum_equation)
    return ars.draw(size)


# def draw_gamma_ras(a, theta, size=1):
#     """
#     returns Gamma distributed samples according to the Rasmussen (2000) definition
#     """
#     return gamma.rvs(0.5 * a, loc=0, scale=2.0 * theta / a, size=size)


def draw_gamma(a, theta, size=1):
    """
    returns Gamma distributed samples
    """
    return gamma.rvs(a, loc=0, scale=theta, size=size)


def draw_invgamma(a, theta, size=1):
    """
    returns inverse Gamma distributed samples
    """
    return invgamma.rvs(a, loc=0, scale=theta, size=size)


def draw_wishart(df, scale, size=1):
    """
    returns Wishart distributed samples
    """
    return wishart.rvs(df=df, scale=scale, size=size)


def draw_normal(loc=0, scale=1, size=1):
    '''
    returns Normal distributed samples
    '''
    return norm.rvs(loc=loc, scale=scale, size=size)


def draw_MVNormal(mean=0, cov=1, size=1):
    """
    returns multivariate normally distributed samples
    """
    return mv_norm.rvs(mean=mean, cov=cov, size=size)


def draw_alpha(k, N, size=1):
    """
    draw alpha from posterior (depends on k, N), eq 15 (Rasmussen 2000), using ARS
    Make it robust with an expanding range in case of failure
    """
    ars = ARS(log_p_alpha, log_p_alpha_prime, xi=[0.1, 5], lb=0, ub=np.inf, k=k, N=N)
    return ars.draw(size)


def draw_indicator(pvec):
    """
    draw stochastic indicator values from multinominal distributions, check wiki
    """
    res = np.zeros(pvec.shape[1])
    # loop over each data point
    for j in range(pvec.shape[1]):
        c = np.cumsum(pvec[ : ,j])        # the cumulative un-scaled probabilities
        R = np.random.uniform(0, c[-1], 1)        # a random number
        r = (c - R)>0                     # truth table (less or greater than R)
        y = (i for i, v in enumerate(r) if v)    # find first instant of truth
        try:
            res[j] = y.__next__()           # record component index
        except:                 # if no solution (must have been all zeros)
            res[j] = np.random.randint(0, pvec.shape[0]) # pick uniformly
    return res


def get_pixel_list(img_list):
    height, width, dim = img_list[0].shape
    pixel_list = np.zeros((height*width, len(img_list), dim))
    for i in range(height):
        for j in range(width):
            for index, img in enumerate(img_list):
                pixel_list[width * i + j][index] = img[i][j]
    return pixel_list

