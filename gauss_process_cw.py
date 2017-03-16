from __future__ import division
import numpy as np
from scipy.optimize import minimize


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:, :-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:, -1])[:, None]
    X = (Xraw - Xraw.mean(axis=0)) / np.std(Xraw, axis=0)
    y = (yraw - yraw.mean(axis=0)) / np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4]  # take every fourth observation for the test set
    train_ind = list(set(ind) - set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test


# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    L = np.linalg.cholesky(cov)  # calculate the cholesky decomposition of the VCV matrix
    Z = np.random.standard_normal(len(mean))  # vector of iid std normal draws (~ N(x|0,1))
    sample = mean + np.inner(L, Z)  # sample is just the mean plus random variance based on the vcv
    return sample


# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2 * self.ln_sigma_f)
        self.sigma2_n = np.exp(2 * self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2 * self.ln_sigma_f)
        self.sigma2_n = np.exp(2 * self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        print(np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n]))
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        print(np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n]))
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0] + Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X = X_aug

        n = X.shape[0]

        covMat = np.zeros((n, n))

        # calculate the kernel for each covmat(i,j) according to data in X, using log params
        l_term = -1.0 / (2 * (self.length_scale ** 2))
        for i in range(n):
            for j in range(n):
                covMat[i][j] = self.sigma2_f * np.exp(
                    l_term * np.inner(X[i] - X[j], X[i] - X[j]))  # X[i] is a row vector

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n * np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        VCV = self.k.covMatrix(self.X, Xa)

        # the final matrix is y f* (ie: we have noisy training observations and non noisy test predictions)

        if self.k.sigma2_n is not None:
            for i in range(self.n, Xa.shape[0] + self.n):
                VCV[i][i] = VCV[i][i] - self.k.sigma2_n

        # decompose the conditional VCV matrix
        K_noisy = VCV[0:self.n, 0:self.n]
        K_ax = VCV[self.n:, 0:self.n]
        K_aa = VCV[self.n:, self.n:]
        K_xa = VCV[0:self.n, self.n:]

        mean_fa = np.dot(np.dot(K_ax, np.linalg.inv(K_noisy)), self.y)

        cov_fa = K_aa - np.dot(np.dot(K_ax, np.linalg.inv(K_noisy)), K_xa)

        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        # breaking this in to two lines for length
        t1 = .5 * np.dot(np.dot(self.y.transpose(), np.linalg.inv(self.K)), self.y) + .5 * np.log(np.linalg.det(self.K))
        lml = t1 + .5 * self.n * np.log(np.pi * 2)

        return lml

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        old_params = self.k.getParamsExp()
        K_bar = self.K - (old_params[2] * np.identity(self.n))

        # reduce K to get the second term from the chain rule
        K_2 = np.log(K_bar) - np.log(old_params[0])
        K_3 = -1 * K_2 * 2 * ((old_params[1]) ** 2)

        # each of the following is dK/dtheta
        dK_sf = 2 * K_bar
        dK_l = K_bar * K_3
        dK_sn = old_params[2] * 2 * np.identity(self.n)

        alpha = np.dot(np.dot(np.linalg.inv(self.K), self.y), ((np.dot(np.linalg.inv(self.K), self.y)).transpose()))

        def grad_theta(dk_dTheta, a, K1):
            return (-1 / 2) * (np.trace(np.dot(a - np.linalg.inv(K1), dk_dTheta)))

        grad_ln_sigma_f = grad_theta(dK_sf, alpha, self.K)
        grad_ln_length_scale = grad_theta(dK_l, alpha, self.K)
        grad_ln_sigma_n = grad_theta(dK_sn, alpha, self.K)

        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        ylen = ya.shape[0]
        mse = 0
        for i in range(ylen):
            mse += (ya[i] - fbar) ** 2
        print(mse)
        print(mse / ylen)
        m = mse / ylen
        return m[0]

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        ylen = ya.shape[0]
        msll = 0
        sig = cov[0][0] + self.k.getParamsExp()[2]
        for i in range(ylen):
            print(ya[i])
            diff = ya[i] - fbar
            msll += (.5 * np.log(2 * np.pi * sig)) + (diff ** 2) / (2 * sig)
        res = (1 / ylen) * msll
        return res[0]

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params=None, disp=True):
        p = np.array([.5 * np.log(1), np.log(.1), .5 * np.log(.5)])
        res = minimize(self.logMarginalLikelihood, p, method='BFGS', jac=self.gradLogMarginalLikelihood,
                       options={'disp': disp})
        print(res)
        print(res.x)
        return res.x


if __name__ == '__main__':
    np.random.seed(42)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
