# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:39:52 2016

@author: Christian W
"""

#Regression Data
import math
import plotly.plotly as py
import plotly.graph_objs as g_o
import numpy as np 
#py.tools.set_credentials_file(username='Fczetndh1', api_key='pmkq3tm1m2')
py.sign_in("Fczetndh1", "pmkq3tm1m2")
import matplotlib.pyplot as plt


#Helper Functions
def factorial(n):
    if n < 0:
        return(0)
    if n == 0:
        return(1)
    else:
        return(n * factorial(n -1))
        

def nChooseK(n,k):
    try:
        nK = factorial(n) / ((factorial(n - k) * factorial(k)))
        return(nK)  
    except(ZeroDivisionError):
        return(0)
  
    

def binom(nTotal, nWhite, p):
    nK = nChooseK(nTotal, nWhite)
    if nK < 1:
        return 0
    return(nK* (p ** nWhite) * ((1-p) **(nTotal - nWhite)))

    
#(1) Discrete Models

#1A
def evidenceD(nTotal, nWhite):
    prob = 0
    for p in range(11):
        prob += (1/11)* binom(nTotal, nWhite, p/10)
    return(prob)
    
def DPP(nTotal, nWhite):  
    postProb = np.zeros(11)
    prior = 1/11
    for i in range(11):
        liklihood = binom(nTotal, nWhite, i/10)
        #liklihood = nChooseK(nTotal, nWhite) * nChooseK(i, nWhite)
        postProb[i] = liklihood * prior / evidenceD(nTotal, nWhite)
    return(postProb)

     
    
    
#1B

def evidenceC(nTotal, nWhite):
    prob = 0
    for p in range(2, 11, 2):
        prob += (1/6)* binom(nTotal, nWhite, p/10)
    return(prob)



#1C

def allEvidence(n):
    allProbC = np.zeros(n+1)
    allProbD = np.zeros(n+1)
    allProbB = np.zeros(n+1)
    for i in range(n+1):
        allProbB[i] = .5*binom(n,i,.5)        
        allProbC[i] = evidenceC(n, i)
        allProbD[i] = evidenceD(n , i)
    allProbB[0] += .25
    allProbB[n] += .25
    return[allProbB, allProbC, allProbD]
c = allEvidence(20)
print(c[0])
print(c[1])
print(c[2])



#Graphs
x = np.arange(21)
print(x)
y1= c[0]
y2 = c[1]
y3 = c[2]
plt.scatter(x, y1, alpha=0.5)
plt.show()
plt.scatter(x, y2, alpha=0.5)
plt.show()
plt.scatter(x, y3, alpha=0.5)
plt.show()


#1D
#Functions to Calculate Posterior for Beth and Charlotte

def BethPP(nTotal, nWhite):
    postProb = np.zeros(11)
    for i in range(11):
        liklihood = binom(nTotal, nWhite, .5) 
        evidence = .5* binom(nTotal, nWhite, .5)
        prior = 0
        if (i == 5):
            prior = .5
        if (i == 0 or i == 10 and nTotal != nWhite):
            evidence += .25
            liklihood = 0 
        postProb[i] = (liklihood * prior) / (evidence)
    return postProb

def CPP(nTotal, nWhite):
    postProb = np.zeros(11)
    evidence = evidenceC(20,13)
    priors = [1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6]    
    for i in range(11):
        prior = priors[i]
        liklihood = binom(nTotal, nWhite, i/10)
        postProb[i] = liklihood * prior / evidence
    return postProb

cpostprob = CPP(20,13)
bpostprob = BethPP(20,13)
dpostprob = DPP(20,13)

    
#1E



#(2) Differentiation

#2C

B = np.array([[3 , -1], [-1, 3]])
a = np.array([1,0])
b = np.array([0,-1])
a.reshape((2,1))
b.reshape((2,1))
B.reshape((2,2))

def f2(x):
    return(np.sin(np.dot((x-a).transpose(), (x-a))) + np.dot(np.dot((x-b).transpose(), B), (x-b)))
    
def f3(x):
    return((1 - (np.exp(-np.dot((x-a).transpose(), (x-a))) + 
     np.exp(-np.dot(np.dot((x-b).transpose(),B),(x-b))) -
     .1*np.log(abs(max((np.dot(x, x.transpose()) +
     np.array([[.01, 0], [0, .01]]).reshape((2,2))))).sum(0)))))
     
    

def grad_f1(x):
    return(x.transpose() + np.dot(x.transpose(), B) + a.transpose() + b.transpose())

def grad_f2(x):
    t1 = np.dot(4, np.dot(np.cos(np.dot((x-a).transpose(), (x-a))), (x-a).transpose()))
    t2 = np.dot(2, np.dot((x-b).transpose(), B))
    return(t1 + t2)

def grad_f3(x):
    xxt = (np.dot(x, x.transpose()) + np.array([[.01, 0], [0, .01]]).reshape((2,2)))
    t1 = 2*np.dot((x-a).transpose(), np.exp(-np.dot((x-a).transpose(), (x-a))))
    t2 = 2 * np.dot(np.dot((x-b).transpose(), B), np.exp(2 * np.dot(-np.dot((x-b).transpose(),B),(x-b))))
    t3 = -np.dot(.1* (2/ np.log(max(xxt.sum(0)))), x.transpose()) 
    return(t1 + t2 + t3)


#2d 

def gradDescent(x, gamma, iterations, gradcalc, f):
    x.reshape((2,1))
    count = 0
    X1 = np.zeros(iterations)
    X2 = np.zeros(iterations)
    Z = np.zeros(iterations)
    while count < iterations:
        temp = x - (gamma* gradcalc(x))
        x1 = temp
        error = x1 - x
        print((((x1[0] - x[0])**2)**.5) + (((x1[1] - x[1])**2)**.5))
        print(x1)
        x = x1
        count += 1
        X1[count -1] = x[0]
        X2[count-1] = x[1]
        Z[count-1]= f(x)
    return([X1,X2,Z])

g2 = gradDescent(np.array([1,-1]), .05, 50, grad_f2 , f2)
g3 = gradDescent(np.array([1,-1]), .0001, 50, grad_f3, f3)

#contour plots

(X, Y) = np.meshgrid(g2[0], g2[1])
Z = np.sin(X**2 - 2*X + 1 + Y**2) + 3*(X**2) + 3*(Y**2) - 2*X*Y - 6*Y + 2*X + 3
#Z = g2[2]
plot1 = plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')

for i in [.1, .2, .3, .5, .7, 1]:
    g2 =gradDescent(np.array([1,-1]), i, 50, grad_f2 , f2)
    #g3 =gradDescent(np.array([1,-1]), i, 50, grad_f3, f3)



#Problem 4
def MLE(phi, y, n):
    w = np.dot(np.dot(np.linalg.inv((np.dot(phi.transpose(), phi))), phi.transpose()),y)
    sigma = 1/n*  np.dot((y - np.dot(phi, w)).transpose(),(y - np.dot(phi, w) ))
    return([w, sigma])


def poly(x, degree):
    phiarray = np.zeros(shape = (len(x), degree + 1))
    phiarray[:,0] = 1
    for i in range(len(x)):
        for a in range(degree + 1):
            phiarray[i][a] = x[i]**a
    return(phiarray)
    
def trig(x, degree):
    phiarray = np.zeros(shape = (len(x), 2* degree + 1))
    phiarray[:,0]= 1
    for i in range(len(x)):
        for a in range(1, 2*degree + 1, 2):
            phiarray[i][a] = np.sin(2* np.pi * a * x[i])
            phiarray[i][a+1] = np.cos(2* np.pi * a * x[i])
    return(phiarray)
    
#4a
N = 25 
X = np.reshape(np.linspace(0, 0.9, N), (N, 1)) 
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

#poly feature matrices
phi0 = poly(X, 0)
phi1 = poly(X, 1)
phi2 = poly(X, 2)
phi3 = poly(X, 3)
phi11 = poly(X, 11)

#optimal poly parameters
params = [MLE(phi0, Y, N), MLE(phi1, Y, N), MLE(phi2, Y, N), MLE(phi3, Y, N), MLE(phi11, Y, N)]

#4a data
predmean = [np.dot(phi0, params[0][0]) + params[0][1],np.dot(phi1, params[1][0]) + params[1][1],
           np.dot(phi2, params[2][0]) + params[2][1],  np.dot(phi3, params[3][0]) + params[3][1],
           np.dot(phi11, params[4][0]) + params[4][1]]
     
      
#4a plotted functions and legend      

fig =plt.figure()
plt.plot(X, predmean[0], label = "Poly function, order = 0") 
plt.plot(X, predmean[1], label = "Poly function,order = 1")  
plt.plot(X, predmean[2], label = "Poly function,order = 2")  
plt.plot(X, predmean[3], label = "Poly function,order = 3") 
plt.plot(X, predmean[4], label = "Poly function,order = 11")      
plt.xlim(-.3, 1.3)
plt.title('Regression of X on Y with Polynomial Basis Functions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
fig.savefig('Poly functions order 0-11', bbox_inches='tight')
plt.close(fig)    
#4B 

#trigonometric feature matrices
tphi1 = trig(X, 1)
tphi11= trig(X, 11)

#optimal trig parameters
paramsTrig = [MLE(tphi1, Y, N), MLE(tphi11, Y, N)]

#4b data
predmeanTrig = [np.dot(tphi1, paramsTrig[0][0]) + paramsTrig[0][1], np.dot(tphi11, paramsTrig[1][0]) + paramsTrig[1][1]]
     
      
#4b plotted functions and legend 
fig =plt.figure()          
plt.plot(X, predmeanTrig[0], label = "Trig Function, order = 1") 
plt.plot(X, predmeanTrig[1], label = "Trig Function, order = 11")  
plt.xlim(-1, 1.2)
plt.ylim(-1, 1.2)
plt.title('Regression of X on Y with Trig Basis Functions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
fig.savefig('Poly functions 1 and 11', bbox_inches='tight')
plt.close(fig)  

#4c 
def testerror(X,Y,N):
    testErrorMat = []
    mleSigma = []
    
    def cVError(X, Y, order, N):
        totalerr = 0
        for i in range(len(X)):
            temp = X
            validSet = X[i]
            temp = np.delete(temp, i)
            tempy = np.delete(Y,i)
            phiMat = trig(temp, order)
            params = MLE(phiMat, tempy, N -1)
            testMat = trig(validSet, order)
            y_hat = np.dot(testMat, params[0]) + params[1]
            totalerr += math.sqrt((Y[i] - y_hat[0])**2)
        return([totalerr/25,params[1]])
    
    for a in range(11):
        testErrorMat.append(cVError(X,Y,a,N)[0])
        mleSigma.append(cVError(X,Y,a,N)[1])
    return([testErrorMat, mleSigma])
        
        
tError = testerror(X,Y,25)

fig =plt.figure()
plt.plot(np.arange(11), tError[0], label = "Test Error as a function of order of basis") 
plt.plot(np.arange(11), tError[1], label = "Variance of Training set as a function of order of basis") 
#plt.plot(X, predmeanTrig[1], label = "Trig Function, order = 11")  
plt.title('LOOCVd mean test erro as function of order of basis')
plt.xlabel('Order of Basis')
plt.ylabel('Y')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
fig.savefig('Test Error as a function of order of basis', bbox_inches='tight')
plt.close(fig)  

#5B

def gaussian(x, number, scale):
	phiMat = np.zeros(shape = (len(x), number + 1))
	phiMat[:,0] = 1
	for i in range(len(x)):
          for a in range(number + 1):
              mean = a/20
              phiMat[i][a] = np.exp(-(x[i] - mean)**2 / (2*(scale**2)))
	return(phiMat)

def regMLE(phiMat, Y, Lambda):
	t1 = np.linalg.inv((np.dot(phiMat.transpose(), phiMat) + (Lambda * np.identity(phiMat.shape[1]))))
	t2 = np.dot(phiMat.transpose(), Y)
	return(np.dot(t1,t2))

gaussMat = gaussian(X, 20, .1)
predGFunc = []

for lambdas in [.001,1, 1000000]:
	gaussBetas = regMLE(gaussMat, Y, lambdas)
	predGFunc.append(np.dot(gaussMat, gaussBetas))
fig =plt.figure()
plt.plot(X, predGFunc[0], label = 'lambda = 1 x 10^ -13')
plt.plot(X, predGFunc[1], label = 'lambda = 100')
plt.plot(X, predGFunc[2], label = 'lambda = 1 x 10 ^ 13')
plt.title('regularized ridge regression predicted function')
plt.xlabel('X value')
plt.ylabel('Regularized Regression Predicted Y')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
fig.savefig('regularized ridge regression predicted function', bbox_inches='tight')
plt.close(fig)

#5C

def make_sample_dataset(): 
    N = 25 
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1)) 
    noise = np.reshape(np.random.randn(X.size), X.shape)
    Y = np.cos(10*X**2)+ 0.2 * (2**-0.5) * noise 
    return Y
    

def bVtrade(dim, Lambda):
    norm = 1/ dim
    X = np.reshape(np.linspace(0, 0.9, 25), (25, 1)) 
    ytrue = np.cos(10 * (X**2))
    phiMat = gaussian(X, 20, .1)
    predYMat= []
    runningvar = 0
    error = 0
    for i in range(dim):
        Y = make_sample_dataset()
        predY = np.dot(phiMat,regMLE(phiMat, Y, Lambda))
        #print(predY)
        predYMat.append(predY) 
        error += np.sum((predY - ytrue)**2)
    
    yavg = np.mean(predYMat, axis = 0)
    #print(ytrue)
    #print("cock")
    #print(predYMat)
    #print(sum((yavg - ytrue)**2))
    bias2 = .04 *  sum((yavg - ytrue)**2)
    
    for a in range(dim):
           runningvar += .04* np.sum((predYMat[a] - yavg)**2)
           #print(runningvar)
    runningvar = norm * runningvar
    error = norm * error
    return([bias2, runningvar, (error**2)**2])
        

lambdas = np.arange(0.000001,.5,.02)

#[.000000001, .0000001, .000001, .0001, .001, .01, .1,.11,.15,.2]
#lambdas = [.01, 1 , 2, 3, 5, 10, 12, 14, 16, 20, 25]
bVmat = []
b2Mat = []
varMat = []
mseMat = []
for Lambda in lambdas:
    result = bVtrade(20, Lambda)
    bVmat.append(result)
    b2Mat.append(result[0])
    varMat.append(result[1])
    mseMat.append(result[2])
    
fig = plt.figure()
plt.plot(lambdas, b2Mat, label = 'Bias squared as a function of lambda')
plt.plot(lambdas, varMat, label = 'variance as a function of lambda')
plt.plot(lambdas, mseMat, label = 'Squared squared error as a function of lambda')
plt.title('Error as a function of Lambda')
plt.xlabel('Value of Regularizer (Lambda)')
plt.ylabel('test error')
plt.ylim(0, .01)
#plt.xlim(0, 25)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
fig.savefig('Bias squared as a function of lambda', bbox_inches='tight')
plt.close(fig)
#6

#6a 
def lml(alpha, beta, Phi, Y):
    n = len(Y)
    covMat = (np.dot(np.dot(Phi, np.identity(Phi.shape[1]) * beta), Phi.transpose()) +
    np.identity(Phi.shape[0]) * alpha)
    detVar = np.linalg.det(covMat)
    t1 = -(n/2)*np.log(2 * np.pi)
    t2 = -(n/2) * np.log(detVar)
    t3 = -.5*np.dot(np.dot(Y.transpose(), np.linalg.inv(covMat)), Y)
    return(-t2 - t3)

def grad_lml(alpha, beta, Phi, Y) :
    covMat = (np.dot(np.dot(Phi, np.identity(Phi.shape[1]) * beta), Phi.transpose()) +
    np.identity(Phi.shape[0]) * alpha)
    gamma = sum((np.linalg.eigvalsh(covMat) - alpha)/(np.linalg.eigvalsh(covMat)))
    dalpha = 1 / sum(1 / np.linalg.eigvalsh(covMat))
    dbeta = 1 / np.dot(np.dot((1 / (25 -  gamma)), Y.transpose()),Y)
    #t1 = np.dot(np.linalg.inv(covMat), Y)
    #t2 = .5 * np.trace(np.dot(t1, t1.transpose()) - np.linalg.inv(covMat))
    #dalpha = t2
    #dbeta = t2 * np.dot(Phi,Phi.transpose())
    #print(dalpha)
   #print(dbeta)
    return(np.array([dalpha, dbeta]))




#6B
grad_lml(1, .1, phi1, Y)

def gradDescentBayes(alpha, beta, Phi, Y, iterations, learn):
    alpha = 1
    beta = .1
    count = 0
    while count < iterations:
        params = grad_lml(alpha, beta, Phi, Y)
        dalpha, dbeta = params[0], params[1]
        tempa = alpha - learn*dalpha
        tempb = beta - learn*dbeta
        while (abs(tempa)>abs(alpha) or abs(tempb) > abs(beta)):
            learn = .5*learn
            tempa = alpha - learn*dalpha
            tempb = beta - learn*dbeta            
        alpha = alpha - learn * dalpha
        beta = beta - learn*dbeta
        count += 1
    return([alpha,beta])
        
c = gradDescentBayes(1, .1, phi1, Y, 100, .5)


#6C
trigHyperMat = []
bayesMLEMat = []
for a in range(13):
    trigHyperMat.append(gradDescentBayes(2, 1, trig(X,a), Y, 100, .001))
    bayesMLEMat.append(lml(trigHyperMat[a][0], trigHyperMat[a][1], trig(X,a), Y))
    
bayesY = []
for a in bayesMLEMat:
    for b in a:
        bayesY.append(b)    
        
fig = plt.figure()        
plt.plot(np.arange(13), bayesY, label = 'Max Marginal Log Likelihood as function of order of basis')
plt.title('Max Marginal LL as function of order of basis')
plt.xlabel('Order of Basis Functions')
plt.ylabel('Max Marginal LL')
#plt.ylim(0, .01)
#plt.xlim(0, 25)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
fig.savefig('Max Marginal Log Likelihood as function of order of basis', bbox_inches='tight')
plt.close(fig)
