import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y): # problem 1 (akannan4)
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
   
    # Handouts B.3 page 6      

    N,d = X.shape
    labels = y.reshape(y.size)
    classes = np.unique(labels)
    no_of_classes = classes.shape[0]
    k = no_of_classes

    means = np.zeros((d, k))

    for cl in range(k):
        XTarget = X[labels == classes[cl]]
        means[:, cl] = np.mean(XTarget, axis=0)

    covmat = np.cov(np.transpose(X))

    return means,covmat

def qdaLearn(X,y): # problem 1 (akannan4)
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # Handouts B.3 page 6

    N,d = X.shape
    labels = y.reshape(y.size)
    classes = np.unique(labels)
    no_of_classes = classes.shape[0]
    k = no_of_classes

    means = np.zeros((d, k))
    covmats = [np.zeros((d,d))]* k

    for cl in range(k):
        XTarget = X[labels == classes[cl]]
        means[:, cl] = np.mean(XTarget, axis=0)
        covmats[cl] = np.cov(np.transpose(XTarget))

    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest): # problem 1 (akannan4)
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # Handouts B.3 page 6

    N,d = Xtest.shape
    _,k = means.shape
    det_covmat = np.linalg.det(covmat)
    inv_covmat = np.linalg.inv(covmat)

    p = np.zeros((N, k))

    for cl in range(k):
        divisor = (np.power(np.pi * 2,d/2) * (np.power(det_covmat, 0.5)))
        c1 = Xtest - means[:, cl]
        c2 = np.dot(inv_covmat, c1.T)
        c3 = np.sum(c1 * c2.T, axis = 1)
        dividend = np.exp(-0.5 * c3)
        p[:, cl] = dividend / divisor

    ypred = np.argmax(p, 1)
    ypred += 1
    ytest = ytest.reshape(ytest.size)

    acc = np.mean(ypred == ytest) * 100

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest): # problem 1 (akannan4)
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # Handouts B.3 page 6

    N,d = Xtest.shape
    _,k = means.shape


    p = np.zeros((N, k))

    for cl in range(k):
        det_covmat = np.linalg.det(covmats[cl])
        inv_covmat = np.linalg.inv(covmats[cl])
        divisor = (np.power(np.pi * 2,d/2) * (np.power(det_covmat, 0.5)))
        c1 = Xtest - means[:, cl]
        c2 = np.dot(inv_covmat, c1.T)
        c3 = np.sum(c1 * c2.T, axis = 1)
        dividend = np.exp(-0.5 * c3)
        p[:, cl] = dividend / divisor

    ypred = np.argmax(p, 1)
    ypred += 1
    ytest = ytest.reshape(ytest.size)

    acc = np.mean(ypred == ytest) * 100

    return acc,ypred

def learnOLERegression(X,y): # problem 2 (arjunsun)
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                

    # w = (X^T.X)^(-1).(X^T.Y)  

    Xtran=np.transpose(X)
    w=np.linalg.inv(np.dot(Xtran,X))
    w=np.dot(np.dot(w,Xtran),y)
    
    return w

def learnRidgeRegression(X,y,lambd): # problem 3 (arjunsun)
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                


    # w = (((X^T.X) + lambd * identity(d)) ^ -1 ).(X^T.Y)

    Xtran=np.transpose(X)
    i=np.identity(X.shape[1])
    w=np.linalg.inv(np.dot(Xtran,X)+np.dot(lambd,i))
    w=np.dot(np.dot(w,Xtran),y)

    return w


def testOLERegression(w,Xtest,ytest): # problem 2 (akannan4)
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    

    # RMSE = (SQRT(SUM((ytest^T - (w^T.Xtest^T))^2)))/N

    wTran=np.transpose(w)
    Xtran=np.transpose(Xtest)
    yTran=np.transpose(ytest)
    N=Xtest.shape[0]
    rmse=np.sum(np.square(np.subtract(yTran,np.dot(wTran,Xtran))))
    rmse=np.sqrt(rmse/N)
    #rmse=rmse/N
    return rmse

def regressionObjVal(w, X, y, lambd): # problem 4 (sammok)

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p): # problem 5 (sammok)
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                   
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
"""
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,pred = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,pred = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

#plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
"""
# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,X_i,y)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3_train)
plt.show()
"""
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
"""
