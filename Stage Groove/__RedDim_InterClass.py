# -*- coding: utf-8 -*-
#from pyriemann.classification import MDM, TSclassifier
# from pyriemann.estimation import Covariances
# from sklearn.pipeline import Pipeline
# from mne.decoding import CSP
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.linalg import eigvalsh

from __ExtractDyn import *

def def_cov_matrix(d0, d1, d2):
	cov0 = np.cov(np.transpose(d0))
	cov1 = np.cov(np.transpose(d1))
	cov2 = np.cov(np.transpose(d2))
	return cov0, cov1, cov2 

def extract_badChann(dd0, dd1, dd2):
	lenght = len(dd0)
	
	m0 = dd0['var'].mean()
	m1 = dd1['var'].mean()
	m2 = dd2['var'].mean()
	
	if (m0 > m1) and (m0 > m2):
		idx = list(dd0.sort_values(by = 'var', ascending = False).index)[0:lenght-10]
	if (m1 > m0) and (m1 > m2):
		idx = list(dd1.sort_values(by = 'var', ascending = False).index)[0:lenght-10]
	if (m2 > m0) and (m2 > m1):
		idx = list(dd2.sort_values(by = 'var', ascending = False).index)[0:lenght-10]
	
	print("Idx Good chann : ", idx)
	return idx

def extract_subdict(idxGoodChan, idx_LobeRchan):
	idx_lobes = idx_LobeRchan.copy()
	for key, value in idx_LobeRchan.items():
		if key in idxGoodChan:
			pass
		else:
			del idx_lobes[key]
	return idx_lobes
			
		

'''
def distance_riemann(covA, covB):
    """
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B
    """
    return np.sqrt((np.log(eigvalsh(covA, covB))**2).sum())

def distance_euclid(A, B):
    """
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B
    """
    return np.linalg.norm(A - B, ord='fro')

def pairwise_distance(X, Y=None, metric='riemann'):
    """Pairwise distance matrix
    :param A: fist Covariances instance
    :param B: second Covariances instance (optional)
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distances between pairs of elements of X or between elements
    of X and Y.
    """
    Ntx, _ = X.shape

    if Y is None:
        dist = np.zeros((Ntx, Ntx))
        for i in range(Ntx):
            for j in range(i + 1, Ntx):
                dist[i, j] = distance(X[i], X[j], metric)
        dist += dist.T
    else:
        Nty, _= Y.shape
        dist = np.empty((Ntx, Nty))
        for i in range(Ntx):
            for j in range(Nty):
                dist[i, j] = distance(X[i], Y[j], metric)
    return dist

def distance(A, B, metric='riemann'):
    """Distance between two covariance matrices A and B according to the
    metric.
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
        'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
        'kullback_sym'.
    :returns: the distance between A and B
    """
	
    distance_methods = {'riemann': distance_riemann}
	
    if callable(metric):
        distance_function = metric
    else:
        distance_function = distance_methods[metric]

    if len(A.shape) == 3:
        d = np.empty((len(A), 1))
        for i in range(len(A)):
            d[i] = distance_function(A[i], B)
    else:
        d = distance_function(A, B)

    return d



def extract_features(X, y):

	cov_data_X = Covariances(estimator='cov').transform(X) #lwf, cov, scm, oas, corr
	# cov_data_test = Covariances().transform(test_X)
	cv = KFold(n_splits=10, random_state=42, shuffle=True)
	
	# clf = TSclassifier()
	# scores = cross_val_score(clf, cov_data_train, train_y, cv=cv, n_jobs=1)
	# print("Tangent space Classification accuracy: ", np.mean(scores))

	#clf = TSclassifier()
	#clf.fit(cov_data_train, train_y)
	#print(clf.score(cov_data_test, test_y))

	mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
	scores = cross_val_score(mdm, cov_test, y, cv=cv, n_jobs=1)
	print("MDM Classification accuracy: ", np.mean(scores))
	mdm = MDM()
	mdm.fit(cov_data_X, y)

	fig, axes = plt.subplots(1, 2)
	ch_names = [ch for ch in range(8)]

	df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
	g = sns.heatmap(
		df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
	g.set_title('Mean covariance - feet')

	df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
	g = sns.heatmap(
		df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
	plt.xticks(rotation='vertical')
	plt.yticks(rotation='horizontal')
	g.set_title('Mean covariance - hands')

	# dirty fix
	plt.sca(axes[0])
	plt.xticks(rotation='vertical')
	plt.yticks(rotation='horizontal')
	plt.savefig("meancovmat.png")
	plt.show()
'''

def extract_GoodSample(X, y):
	pass

if __name__ == '__main__':
	
	X,y,idx_LobeRchan,_,_  = subsample(isuj = 18, r = 2, lobe ="LFidx", moy = True, f = 6, lengc = 0) 
	y = trans_y(y, 2)
	
	# Retourne les indices des essais selon la classes 
	df_count, dict_idx = desc_y(y)
	
	# Retourne les sous-échantillons X par classe
	X = StandardScaler().fit_transform(X)
	X_c0, X_c1, X_c2 = desc_X(X, dict_idx)

	# Retourne les sous-échantillons précédent moyennés 
	dd0, dd1, dd2 = samplecatX(X_c0, X_c1, X_c2)
	
	# Transforme l'index réel en index théorique 
	idxRT = tradix(idxR, idxT)
	lidxTh = idx_change_toReal(idx_LobeRchan, idxRT)
	
	cov0, cov1, cov2 = def_cov_matrix(X_c0, X_c1, X_c2)
	idxGoodChan = extract_badChann(dd0, dd1, dd2)
	idx_lobeRchan2 = extract_subdict(idxGoodChan, idx_LobeRchan)
	
	
	plt.figure(figsize=(10,10))
	sns.set(font_scale=1.5)
	hm = sns.heatmap(cov2,
                 cbar=True,
                 annot=False,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 cmap='coolwarm')
	plt.title('Covariance matrix', size = 18)
	plt.tight_layout()
	plt.show()

	print(dd0['var'].mean())
	print(dd1['var'].mean())
	print(dd2['var'].mean())

	
	# dr = distance_riemann(cov2, cov0)
	# prd = pairwise_distance(cov0, cov2, metric = "riemann")