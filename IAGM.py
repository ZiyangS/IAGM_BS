import os
import time
import copy
import numpy as np
import cv2 as cv
from numpy.linalg import inv, norm, det, slogdet
from utils import *
from scipy.sparse import *
from numba import jit, njit, autojit, vectorize, guvectorize, float64, float32

@jit(nogil=True,)
def check(pixel, mu, s_l, s_r, k=4):
    '''
    check whether a pixel match a Gaussian distribution. Matching means pixel is less than
    2.5 standard deviations away from a Gaussian distribution.
    '''
    lnorm = np.linalg.norm(np.sqrt(inv(np.diag(s_l))))
    rnorm = np.linalg.norm(np.sqrt(inv(np.diag(s_r))))
    x = np.mat(np.reshape(pixel, (3, 1)))
    mu = np.mat(mu).T
    distance = np.linalg.norm((x - mu))
    if (distance < k * lnorm) or (distance < k * rnorm):
        # return True
        return distance
    else:
        return False
    # x = np.mat(np.reshape(pixel, (3, 1)))
    # mu = np.mat(mu).T
    # sigma_l = np.mat(inv(np.diag(s_l)))
    # sigma_r = np.mat(inv(np.diag(s_r)))
    # d_l = np.sqrt((x - mu).T * sigma_l.I * (x - mu))
    # d_r = np.sqrt((x - mu).T * sigma_r.I * (x - mu))
    # if (d_l < k) or (d_r < k):
    #     return True
    # else:
    #     return False


class Asy_Gaussian():
    '''
    the model of a asymmetric gaussian model
    '''
    def __init__(self, mu, s_l, s_r):
        self.mu = mu
        self.s_l = s_l
        self.s_r = s_r

def Asy_Gaussian_pdf(x, mu, s_l, s_r):
    y = np.zeros(3)
    # y = 1
    for i, x_k in enumerate(x):
        if x_k < mu[i]:
            # y *= np.sqrt(2/np.pi)/(np.power(s_l[i], -0.5) + np.power(s_r[i], -0.5))\
            #        * np.exp(- 0.5 * s_l[i] * (x_k - mu[i])**2)
            y[i] = np.sqrt(2/np.pi)/(np.power(s_l[i], -0.5) + np.power(s_r[i], -0.5))\
                   * np.exp(- 0.5 * s_l[i] * (x_k - mu[i])**2)
        else:
            # y *= np.sqrt(2/np.pi)/(np.power(s_l[i], -0.5) + np.power(s_r[i], -0.5))\
            #        * np.exp(- 0.5 * s_r[i] * (x_k - mu[i])**2)
            y[i] = np.sqrt(2/np.pi)/(np.power(s_l[i], -0.5) + np.power(s_r[i], -0.5))\
                   * np.exp(- 0.5 * s_r[i] * (x_k - mu[i])**2)
    return y


class AGD_mixture_model():
    def __init__(self, X):
        N, D = X.shape
        muy = np.mean(X, axis=0)
        vary = np.zeros(D)
        for k in range(D):
            vary[k] = np.var(X[:, k])

        self.c = np.zeros(N)  # initialise the stochastic indicators
        self.pi = np.ones(1)  # initialise the weights
        self.n = np.array([N])  # initialise the occupation numbers
        self.M = 1

        self.mu = np.zeros((1, D))  # initialise the means
        self.s_l = np.zeros((1, D))  # initialise the precisions
        self.s_r = np.zeros((1, D))  # initialise the precisions

        # draw probability from prior
        self.mu[0, :] = muy
        self.beta_l = np.array([np.squeeze(draw_invgamma(0.5, 2)) for d in range(D)])
        self.beta_r = np.array([np.squeeze(draw_invgamma(0.5, 2)) for d in range(D)])
        self.w_l = np.array([np.squeeze(draw_gamma(0.5, 2 * vary[k])) for k in range(D)])
        self.w_r = np.array([np.squeeze(draw_gamma(0.5, 2 * vary[k])) for k in range(D)])
        self.s_l[0, :] = np.array([np.squeeze(draw_gamma(self.beta_l[k] / 2, 2 / (self.beta_l[k] * self.w_l[k])))
                                   for k in range(D)])
        self.s_r[0, :] = np.array([np.squeeze(draw_gamma(self.beta_r[k] / 2, 2 / (self.beta_r[k] * self.w_r[k])))
                                   for k in range(D)])
        self.lam = draw_MVNormal(mean=muy, cov=vary)
        self.r = np.array([np.squeeze(draw_gamma(0.5, 2 / vary[k])) for k in range(D)])
        self.alpha = 1.0 / draw_gamma(0.5, 2.0)
        self.var_l = 1/self.s_l
        self.var_r = 1/self.s_r

    def train(self, X, Nsamples=80):
        # record the start time
        ori_t = time.time()
        N, D = X.shape
        print('{}: initialised parameters'.format(time.asctime()))
        # loop over samples
        z = 1
        oldpcnt = 0
        while z < Nsamples:
            # recompute muy and covy
            muy = np.mean(X, axis=0)
            vary = np.zeros(D)
            for k in range(D):
                vary[k] = np.var(X[:, k])
            precisiony = 1 / vary

            Xj = [X[np.where(self.c == j), :] for j, nj in enumerate(self.n)]
            mu_cache = self.mu
            self.mu = np.zeros((self.M, D))
            j = 0
            # draw muj from posterior (depends on sj, c, lambda, r)
            for x, nj, s_lj, s_rj in zip(Xj, self.n, self.s_l, self.s_r):
                x = x[0]
                for k in range(D):
                    x_k = x[:, k]
                    p = x_k[x_k < mu_cache[j][k]].shape[0]
                    q = x_k[x_k >= mu_cache[j][k]].shape[0]
                    x_l_sum = np.sum(x_k[x_k < mu_cache[j][k]])
                    x_r_sum = np.sum(x_k[x_k >= mu_cache[j][k]])
                    r_n = self.r[k] + p * s_lj[k] + q * s_rj[k]
                    mu_n = (s_lj[k] * x_l_sum + s_rj[k] * x_r_sum + self.r[k] * self.lam[k]) / r_n
                    self.mu[j, k] = draw_normal(mu_n, 1 / r_n)
                j += 1

            # draw lambda from posterior (depends on mu, M, and r)
            mu_sum = np.sum(self.mu, axis=0)
            loc_n = np.zeros(D)
            scale_n = np.zeros(D)
            for k in range(D):
                scale = 1 / (precisiony[k] + self.M * self.r[k])
                scale_n[k] = scale
                loc_n[k] = scale * (muy[k] * precisiony[k] + self.r[k] * mu_sum[k])
            self.lam = draw_MVNormal(loc_n, scale_n)

            # draw r from posterior (depnds on M, mu, and lambda)
            temp_para_sum = np.zeros(D)
            for k in range(D):
                for muj in self.mu:
                    temp_para_sum[k] += np.outer((muj[k] - self.lam[k]), np.transpose(muj[k] - self.lam[k]))
            self.r = np.array([np.squeeze(draw_gamma((self.M + 1) / 2, 2 / (vary[k] + temp_para_sum[k]))) for k in range(D)])

            # draw alpha from posterior (depends on M, N)
            self.alpha = draw_alpha(self.M, N)

            # draw sj from posterior (depends on mu, c, beta, w)
            for j, nj in enumerate(self.n):
                Xj = X[np.where(self.c == j), :][0]
                # for every dimensionality, compute the posterior distribution of s_ljk, s_rjk
                for k in range(D):
                    x_k = Xj[:, k]
                    # p represent the number of x_ik < mu_jk
                    p = x_k[x_k < self.mu[j][k]].shape[0]
                    # q represent the number of x_ik >= mu_jk, q = n - p
                    q = x_k[x_k >= self.mu[j][k]].shape[0]
                    # x_l represents the data from i to n of x_ik, which x_ik < mu_jk
                    x_l = x_k[x_k < self.mu[j][k]]
                    # x_r represents the data from i to n of x_ik, which x_ik >= mu_jk
                    x_r = x_k[x_k >= self.mu[j][k]]
                    cumculative_sum_left_equation = np.sum((x_l - self.mu[j][k]) ** 2)
                    cumculative_sum_right_equation = np.sum((x_r - self.mu[j][k]) ** 2)
                    # def Metropolis_Hastings_Sampling_posterior_sljk(s_ljk, s_rjk, nj, beta, w, sum):
                    self.s_l[j][k] = Metropolis_Hastings_Sampling_posterior_sljk(s_ljk=self.s_l[j][k], s_rjk=self.s_r[j][k],
                                                                            nj=nj, beta=self.beta_l[k], w=self.w_l[k],
                                                                            sum=cumculative_sum_left_equation)
                    self.s_r[j][k] = Metropolis_Hastings_Sampling_posterior_srjk(s_ljk=self.s_l[j][k], s_rjk=self.s_r[j][k],
                                                                            nj=nj, beta=self.beta_r[k], w=self.w_r[k],
                                                                            sum=cumculative_sum_right_equation)

            # draw w from posterior (depends on k, beta, D, sj)
            self.w_l = np.array([np.squeeze(draw_gamma(0.5 * (self.M * self.beta_l[k] + 1), \
                                                       2 / (
                                                       vary[k] + self.beta_l[k] * np.sum(self.s_l, axis=0)[k]))) \
                                 for k in range(D)])
            self.w_r = np.array([np.squeeze(draw_gamma(0.5 * (self.M * self.beta_r[k] + 1), \
                                                       2 / (
                                                       vary[k] + self.beta_r[k] * np.sum(self.s_r, axis=0)[k]))) \
                                 for k in range(D)])

            # draw beta from posterior (depends on k, s, w)
            self.beta_l = np.array([draw_beta_ars(self.w_l, self.s_l, self.M, k)[0] for k in range(D)])
            self.beta_r = np.array([draw_beta_ars(self.w_l, self.s_l, self.M, k)[0] for k in range(D)])

            # compute the unrepresented probability
            p_unrep = (self.alpha / (N - 1.0 + self.alpha)) * integral_approx(X, self.lam, self.r, self.beta_l, self.beta_r,
                                                                              self.w_l, self.w_r)
            p_indicators_prior = np.outer(np.ones(self.M + 1), p_unrep)

            # for the represented components
            for j in range(self.M):
                # n-i,j : the number of oberservations, excluding Xi, that are associated with component j
                nij = self.n[j] - (self.c == j).astype(int)
                idx = np.argwhere(nij > 0)
                idx = idx.reshape(idx.shape[0])
                likelihood_for_associated_data = np.ones(len(idx))
                for i in range(len(idx)):
                    for k in range(D):
                        if X[i][k] < self.mu[j][k]:
                            likelihood_for_associated_data[i] *= 1/(np.power(self.s_l[j][k], -0.5) + np.power(self.s_r[j][k], -0.5))*\
                                                                 np.exp(- 0.5 * self.s_l[j][k] * np.power(X[i][k] - self.mu[j][k], 2))
                        else:
                            likelihood_for_associated_data[i] *= 1/(np.power(self.s_l[j][k], -0.5) + np.power(self.s_r[j][k], -0.5))*\
                                                                 np.exp(- 0.5 * self.s_r[j][k] * np.power(X[i][k] - self.mu[j][k], 2))
                p_indicators_prior[j, idx] = nij[idx] / (N - 1.0 + self.alpha) * likelihood_for_associated_data
            # stochastic indicator (we could have a new component)
            self.c = np.hstack(draw_indicator(p_indicators_prior))

            # sort out based on new stochastic indicators
            nij = np.sum(self.c == self.M)  # see if the *new* component has occupancy
            if nij > 0:
                # draw from priors and increment M
                newmu = np.array([np.squeeze(draw_normal(self.lam[k], 1/self.r[k])) for k in range(D)])
                news_l = np.array([np.squeeze(draw_gamma(self.beta_l[k]/2, 2/(self.beta_l[k] * self.w_l[k]))) for k in range(D)])
                news_r = np.array([np.squeeze(draw_gamma(self.beta_r[k]/2, 2/(self.beta_r[k] * self.w_r[k]))) for k in range(D)])
                self.mu = np.concatenate((self.mu, np.reshape(newmu, (1, D))))
                self.s_l = np.concatenate((self.s_l, np.reshape(news_l, (1, D))))
                self.s_r = np.concatenate((self.s_r, np.reshape(news_r, (1, D))))
                self.M = self.M + 1
                
            # find the associated number for every components
            self.n = np.array([np.sum(self.c == j) for j in range(self.M)])

            # find unrepresented components
            badidx = np.argwhere(self.n == 0)
            Nbad = len(badidx)
            # remove unrepresented components
            if Nbad > 0:
                self.mu = np.delete(self.mu, badidx, axis=0)
                self.s_l = np.delete(self.s_l, badidx, axis=0)
                self.s_r = np.delete(self.s_r, badidx, axis=0)
                # if the unrepresented compont removed is in the middle, make the sequential component indicators change
                for cnt, i in enumerate(badidx):
                    idx = np.argwhere(self.c >= (i - cnt))
                    self.c[idx] = self.c[idx] - 1
                self.M -= Nbad  # update component number

            # recompute n
            self.n = np.array([np.sum(self.c == j) for j in range(self.M)])
            # recompute pi
            self.pi = self.n.astype(float) / np.sum(self.n)
            pcnt = int(100.0 * z / float(Nsamples))
            if pcnt > oldpcnt:
                print('{}: %--- {}% complete ----------------------%'.format(time.asctime(), pcnt))
                oldpcnt = pcnt
            z += 1
            print(self.n)
        # print computation time
        print("{}: time to complete main analysis = {} sec".format(time.asctime(), time.time() - ori_t))


    def continue_train(self, X, train_num=300, beta=0.05):
        match_num = 0
        for pixel in X:
            closest_dist = 10000
            # Check whether match the existing K Gaussian distributions
            match = -1
            for k in range(self.M):
                if check(pixel, self.mu[k], self.s_l[k], self.s_r[k]):
                    if closest_dist > check(pixel, self.mu[k], self.s_l[k], self.s_r[k]):
                        closest_dist = check(pixel, self.mu[k], self.s_l[k], self.s_r[k])
                        match = k
                    match_num += 1
            # a match found
            if match != -1:
                mu = self.mu[match]
                s_l = self.s_l[match]
                s_r = self.s_r[match]
                var_l = self.var_l[match]
                var_r = self.var_r[match]
                x = pixel.astype(np.float)
                delta = x - mu
                self.pi = (1 - beta) * self.pi
                self.pi[match] += beta

                rho = beta * Asy_Gaussian_pdf(pixel, mu, s_l, s_r)
                # rho = pixel * beta
                self.mu[match] = mu + rho * delta
                var_l = np.abs(var_l + rho * (np.matmul(delta, delta.T) - var_l)) + 0.0001
                var_r = np.abs(var_r + rho * (np.matmul(delta, delta.T) - var_r)) + 0.0001
                self.s_l[match] = 1/var_l
                self.s_r[match] = 1/var_r
                # for k in range(3):
                #     if delta[k] < 0:
                #         self.mu[match][k] = mu[k] + beta * (delta[k] * s_l[k])
                #         self.s_l[match][k] = s_l[k] + beta * (0.5 * np.power(s_l[k], -1.5) / (np.power(s_l[k], -0.5)+np.power(s_r[k], -0.5))
                #                                               - 0.5 * (delta[k] ** 2))
                #         self.s_r[match][k] = s_r[k] + beta * (0.5 * np.power(s_r[k], -1.5) / (np.power(s_l[k], -0.5) + np.power(s_r[k], -0.5)))
                #     else:
                #         self.mu[match][k] = mu[k] + beta * (delta[k] * s_r[k])
                #         self.s_l[match][k] = s_l[k] + beta * (0.5 * np.power(s_l[k], -1.5) / (np.power(s_l[k], -0.5)+np.power(s_r[k], -0.5)))
                #         self.s_r[match][k] = s_r[k] + beta * (0.5 * np.power(s_r[k], -1.5) / (np.power(s_l[k], -0.5)+np.power(s_r[k], -0.5))
                #                                               - 0.5 * (delta[k] ** 2))
            # if none of the K distributions match the current value
            # the least probable distribution is replaced with a distribution
            # with current value as its mean, an initially high variance and low rior weight
            if match == -1:
                w_list = [self.pi[k] for k in range(self.M)]
                id = w_list.index(min(w_list))
                # weight keep same, replace mean with current value and set high variance
                self.mu[id] = np.array(pixel).reshape(1, 3)
                self.s_l[id] = np.array([0.03, 0.03, 0.03])
                self.s_r[id] = np.array([0.03, 0.03, 0.03])


class IAGM():
    """
    Class for defining a single sample
    infinite asymmetric gaussian distribution(AGD) mixture model
    """
    def __init__(self, data_dir, results_path, ROI_path, train_num, test_num, init_alpha=0.01):
        self.data_dir = data_dir
        self.results_path = results_path
        self.ROI_path = ROI_path

        self.train_num = train_num
        self.test_num = test_num
        self.img_shape = None

        self.threshold = 0.9
        self.B = 0
        self.alpha = init_alpha
        self.weight_order = None

        self.agd_mat = None
        self.weight = None
        self.mu = None
        self.s_l = None
        self.s_r = None
        self.num_of_mix = 0


    def reorder(self, T=0.90):
        '''
        reorder the estimated components based on the ratio pi / ||σ_lj|| + ||σ_rj||, norm of standard deviation.
        the first B components are chosen as background components
        the default threshold is 0.95
        '''
        k_weight = self.agd_mat.pi
        s_l = self.agd_mat.s_l
        s_r = self.agd_mat.s_r
        k_norm = [np.linalg.norm(np.sqrt(inv(np.diag(s_l[m])))) + np.linalg.norm(np.sqrt(inv(np.diag(s_r[m]))))
                  for m in range(k_weight.shape[0])]
        ratio = k_weight/k_norm
        descending_order = np.argsort(-ratio)
        self.agd_mat.pi = k_weight[descending_order]
        self.agd_mat.mu = self.agd_mat.mu[descending_order]
        self.agd_mat.s_l = s_l[descending_order]
        self.agd_mat.s_r = s_r[descending_order]
        cum_weight = 0
        for index, order in enumerate(descending_order):
            cum_weight += self.agd_mat.pi[index]
            if cum_weight > T:
                self.B = index + 1
                break


    def infer(self, pixel_list):
        '''
        infer whether its background or foregound
        if the pixel is background, both values of rgb will set to 255. Otherwise not change the value
        '''
        result = np.ones((self.test_num + 1 - self.train_num), dtype=int)
        mixture_model = self.agd_mat
        detected_fg_num = 0
        for index, pixel in enumerate(pixel_list):
            for k in range(self.B):
                if check(pixel, mixture_model.mu[k], mixture_model.s_l[k], mixture_model.s_r[k]):
                    result[index] = 0
                    # 0 is black, the background color will be set black
                    break
            if result[index] != 0:
                # foreground pixel will set white, 255
                result[index] = 255
                detected_fg_num += 1
        if detected_fg_num >0:
            print("fg number")
            print(detected_fg_num)
        return result
        

    def model_training(self):
        img_list = []
        results_img_list = []
        # file numbers are from 1 to train_number
        for index in range(self.train_num-1):
            file_name = os.path.join(self.data_dir, 'in%06d'%(index+1) + '.jpg')
            img_list.append(cv.imread(file_name))
        # the shape of results_list is 1200 - 300 , 240, 360, 3
        for index in range(self.test_num):
            if index + 1 < self.train_num:
                continue
            else:
                file_name = os.path.join(self.data_dir, 'in%06d' % (index + 1) + '.jpg')
                results_img_list.append(cv.imread(file_name))
        print(len(img_list))
        print(len(results_img_list))
        self.img_shape = img_list[0].shape
        print(self.img_shape)
        eva_number, i_dict, j_dict = read_ROI(self.ROI_path)
        print(eva_number)
        res_list = np.zeros((len(results_img_list), self.img_shape[0], self.img_shape[1]), dtype=int)
        print(res_list.shape)

        # shape:240*360, 299//10, 3. 240*360 is the pixel shape for a frame, 299//10 is the number of train frames, 3 is RGB value
        pixel_list = get_pixel_list(img_list, eva_number, i_dict, j_dict)
        # shape:240*360, 800//10, 3. 240*360 is the pixel shape for a frame, 800//10 is the number of test frames, 3 is RGB value
        test_pixel_list = get_pixel_list(results_img_list, eva_number, i_dict, j_dict)
        print(pixel_list.shape)
        print(test_pixel_list.shape)

        pixel_zip = zip(pixel_list, test_pixel_list)
        for index, (pixel_data, test_pixel_data) in enumerate(pixel_zip):
            if index == 0:
                print("_____")
                print(index)
                print([i_dict[index]])
                print([j_dict[index]])
                self.agd_mat = AGD_mixture_model(pixel_data)
                self.agd_mat.train(pixel_data)
                # for i in range(self.agd_mat.M):
                #     self.agd_mat.s_l[i] = np.array([30, 30 ,30])
                #     self.agd_mat.s_r[i] = np.array([0.03, 0.03 ,0.03])
                self.reorder()
                self.agd_mat.var_l = 1/self.agd_mat.s_l
                self.agd_mat.var_r = 1/self.agd_mat.s_r
                result_pixel = self.infer(test_pixel_data)
                for i, pixel in enumerate(result_pixel):
                    res_list[i][i_dict[index]][j_dict[index]] = pixel
            else:
                print("_____")
                print(index)
                print([i_dict[index]])
                print([j_dict[index]])
                self.agd_mat.continue_train(pixel_data)
                self.reorder()
                self.agd_mat.var_l = 1 / self.agd_mat.s_l
                self.agd_mat.var_r = 1 / self.agd_mat.s_r
                result_pixel = self.infer(test_pixel_data)
                for i, pixel in enumerate(result_pixel):
                    res_list[i][i_dict[index]][j_dict[index]] = pixel
        for index, res_img in enumerate(res_list):
            cv.imwrite(self.results_path + '/in%06d' % (index + self.train_num) + '.jpg', res_img.reshape(self.img_shape[0:2]))



