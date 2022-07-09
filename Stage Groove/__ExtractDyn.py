# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import kaleido

from statistics import variance
from __Channels_Projection import * 
from __RedDim_InterClass import * 



def extract_data(isuj):
	# Data access
	try : 
		path = 'C:/Users/themi/Data/'
		bad1 = sio.loadmat( str(path) + 'bad.mat') # chan
		bad2 = sio.loadmat(str(path) + 'bad_bn_process.mat'); # trials
		GoodChannel_ImagingKernel = sio.loadmat(str(path) + 'GoodChannel.mat')
		raw = sio.loadmat(str(path) + 'Chan_data_7regr.mat')

	except FileNotFoundError:
		path = '/Users/swann/Université/Stage/Spyder/DNN/Royce decoding/'
		bad1 = sio.loadmat( str(path) + 'bad.mat') # chan
		bad2 = sio.loadmat(str(path) + 'bad_bn_process.mat'); # trials
		GoodChannel_ImagingKernel = sio.loadmat(str(path) + 'GoodChannel.mat')
		raw = sio.loadmat(str(path) + 'Chan_data_7regr.mat')

	GoodChannel = GoodChannel_ImagingKernel['GoodChannel'][:,isuj]
	
	#  bad trials
	goodtrial = np.reshape(bad2['badtrial_bnprocess'][isuj], 144)
	goodtrial = np.where(goodtrial == 1)[0]
	
	#  bad channels
	goodchan = np.zeros((299))
	goodchan[np.concatenate(GoodChannel[0])] = 1
	goodchan = goodchan[bad1['iMeg_allsuj'][0]]
	goodchan = np.where(goodchan == 1)[0]
	
	dictres = {
		'isuj' : isuj,
		'goodtrial' : goodtrial,
		'goodchan' : goodchan
		}
	
	return dictres

def load_alldata():
	'''Returns
	-------
	X : Raw data, features.
	y : Raw data, targets.
	'''
	
	try : 
		path = 'C:/Users/themi/Data/'
		raw = sio.loadmat(str(path) + 'Chan_data_7regr.mat')

	except FileNotFoundError:
		path = '/Users/swann/Université/Stage/Spyder/DNN/Royce decoding/'
		raw = sio.loadmat(str(path) + 'Chan_data_7regr.mat')

	X = raw['Y'] # MEG
	y = raw['X'][:,:,0:3] # 3 regressors of interest: periodicity, syncope, groove
	
	return X, y 


def lobes_idx():
	path = '/Users/swann/Université/Stage/Spyder/Analysis.'
	path_sl = 'sort_loc.csv'
	data_chanReal = pd.read_csv(path_sl)
	path = '4D248.csv'
	data_chanTh = pd.read_csv(path, sep=';')
	data_chanTh = data_chanTh[0:248]

	
	LF = [229, 212, 177, 153, 125, 93, 64, 39, 20, 19, 36, 59, 86, 117, 149, 176, 195,
		  228, 248, 37, 60, 87, 118, 150, 151, 152, 121, 122, 123, 124, 92, 63, 38,
		  61, 88, 119, 120, 89, 90, 91, 62]
	
	LP = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 35,
		  18, 5, 6, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 17, 4, 7, 8,
		  9, 10, 11, 12, 13, 14, 15, 16, 3, 2, 1]
	
	LTG = [65, 66, 67, 68, 69, 70, 71, 95, 96, 97, 98, 99, 100, 126, 127, 128, 129, 
		   130, 131, 132, 154, 155, 156, 157, 158, 159, 160, 178, 179, 180, 181, 
		   196, 197, 198, 199, 230, 231, 232, 233, 234, 214, 215, 213, 94]
	
	LTD = [79, 80, 81, 82, 83, 84, 85, 110, 111, 112, 113, 114, 115, 116, 142, 143, 144,
		   145, 146, 147, 148, 169, 170, 171, 172, 173, 174, 175, 191, 192, 193, 194,
		   225, 226, 227, 244, 245, 246, 247, 243, 208, 209, 210, 211]
	
	LO = [72, 73, 74, 75, 76, 77, 78, 101, 102, 103, 104, 105, 106, 107, 108, 109,
		   133, 134, 135, 136, 137, 138, 139, 140, 141, 161, 162, 163, 164, 165, 166,
		   167, 168, 182, 183, 184, 185, 186, 187, 188, 189, 190, 200, 201, 202, 203, 204, 
		   205, 206, 207, 216, 217, 218, 219, 220, 221, 222, 223, 224, 235, 236, 237,
		   238, 239, 240, 241, 242] 
	
	LFidx, LPidx, LTGidx, LTDidx, LOidx = extract_idx(LF, LP, LTG, LTD, LO, data_chanReal)

	dict_lobes = {
		'LFidx': LFidx,
		'LPidx' : LPidx,
		'LTGidx': LTGidx,
		'LTDidx': LTDidx, 
		'LOidx': LOidx
		}
	
	return dict_lobes

def goodchan(lobe, gc):
	intersection = [x for x in lobe if x in list(gc)]
	return intersection

		
def subsample(isuj, r, lobe, moy, f, lengc, bande):
	''' Parameters
	----------
	isuj : number of subject (int).
	r : regressor (int).
	lobe : lobe (string).
	moy : (bool).
	f : (int).
	lengc : 0 if 1 lobe, else len of each lobe.

	Returns
	-------
	tab : features.
	target : target.
	dictionary : association of index.
	gc : size of tab. '''
	
	lobes = lobes_idx()
	good_idx = extract_data(isuj = isuj)
	gc = goodchan(lobes[lobe], good_idx['goodchan'])
	# gc = lobes[lobe]
	gt = list(good_idx['goodtrial'])
	
	X, y = load_alldata()
	data = X[isuj,gt,:,:]
	data = data[:,gc,:]
	print(X.shape)
	
	if type(r) == list:
		target = y[isuj, gt, :]
	else:
		target = y[isuj, gt, r]
	
	idx = list(range(lengc, lengc + len(gc)))
	dictionary = dict(zip(idx, gc))
	
	if moy == True: # True : return dynamically arrays with 6 frequencies
			dataMoy = data.copy()
			print("Shape Data Before Averaged", dataMoy.shape)
			
			if bande == 'classic':
				bandes = [0, 4, 9, 13, 30, 41, 61, 81, 100]
			if bande == 'log':
				bandes = [0, 1 ,3 ,4, 7, 11, 17, 27, 41, 64, 100]
				
			tab = np.zeros([len(gt), len(gc), len(bandes)-1])
			print("Shape Data after Averaged", tab.shape)
			
			for i in range(1,len(bandes)) :
					tab[:,:, i-1]  = np.mean(dataMoy[:,:, bandes[i-1]:bandes[i]], axis = 2)
					
			tab = tab[:,:,f]
	else: 
		tab = data[:,:,f]
	
	print("Subsample done")
	return tab, target, dictionary, len(gc), gc


def desc_X(X, dict_idx, r): 
	if r == 0:
		X_c0 = X[dict_idx['c0'], :]
		X_c1 = X[dict_idx['c1'], :]
		return X_c0, X_c1
	
	X_c0 = X[dict_idx['c0'], :] ; X_c1 = X[dict_idx['c1'], :] ; X_c2 = X[dict_idx['c2'], :]

	return X_c0, X_c1, X_c2

def desc_y(y,r):
	c0idx = [] ; c1idx = [] ; c2idx = []  
	c0 = 0 ; c1 = 0 ; c2 = 0
	
	for i in range(0, len(y)):
		if y[i] == 0:
			c0idx.append(i)
			c0 += 1
		if y[i] == 1:
			c1idx.append(i)
			c1 += 1
		if y[i] == 2:
			c2idx.append(i)
			c2 += 1

	d = {'Low': [c0], 'Medium': [c1], 'High' : [c2]}
	df_count = pd.DataFrame(data = d)
	dict_idx = {'c0' : c0idx, 'c1' : c1idx, 'c2' : c2idx}
		
	return df_count, dict_idx

def plot_desc_y(y, l, r, isuj):
	reg = ['Periodicity', 'Syncope', 'Groove']
	title = "Distribution of classes for " + str(reg[r])
	
	if type(r) != list:
		if r != 0: 
			fig = px.bar(x=["Low","Medium","High"], y=[y['Low'][0], y['Medium'][0], y['High'][0]], 
					  color = [0, 1,2], title = title, height=800, width=1200)
			fig.add_hline(y=l/3)
			fig.add_hline(y=l/3 + 15, line_dash="dash")
			fig.add_hline(y=l/3 - 15, line_dash="dash")
		else : 
			fig = px.bar(x=["No Periodicity","Periodicity"], y=[y['Low'][0], y['Medium'][0]], 
					  color = [0,1], title = title, height=800, width=1200)
			fig.add_hline(y=l/2)
			fig.add_hline(y=l/2 + 25, line_dash="dash")
			fig.add_hline(y=l/2 - 25, line_dash="dash")
		#fig.write_html('desc_y.html', auto_open=True)
		titlesvg = "barplot" + str(reg[r] + "_p" + str(isuj)) + "_.svg"
		titlepng= "barplot" + str(reg[r] + "_p" + str(isuj)) + "_.png"
		fig.write_image(str(titlesvg))
		fig.write_image(str(titlepng))
		fig.show()
	
	
def samplecatX(X_c0, X_c1, X_c2):
	m0 = np.mean(X_c0[:,:], axis = 0) ; std0 = np.std(X_c0[:,:], axis = 0) ; var0 = np.var(X_c0[:,:],axis = 0)
	m1 = np.mean(X_c1[:,:], axis = 0); std1 = np.std(X_c1[:,:], axis = 0); var1 = np.var(X_c1[:,:],axis = 0)
	m2 = np.mean(X_c2[:,:], axis = 0); std2 = np.std(X_c2[:,:], axis = 0) ; var2 = np.var(X_c2[:,:],axis = 0)

	d0 = {'Mean' : m0, 'std' : std0, 'var' : var0, 'Perception' : 'Low'}
	dd0 = pd.DataFrame(data = d0)
	d1 = {'Mean' : m1, 'std' : std1, 'var' : var1, 'Perception' : 'Medium'}
	dd1 = pd.DataFrame(data = d1)
	d2 = {'Mean' : m2, 'std' : std2, 'var' : var2, 'Perception' : 'High'}
	dd2 = pd.DataFrame(data = d2)
	 
	return dd0, dd1, dd2
 
 
def plot_desc_X(X_c0, X_c1, X_c2, r, lobe, isuj, f, moy):
	'''Parameters
	----------
	X_c0, X_c1, X_c2 : subsamples (arrays)
	r : regressor (int).
	lobe : string.
	isuj : int.
	f : frequency band (0-6) or number's frequency.
	moy : if True, f frequency band.

	Returns
	-------
	Plot descriptives analysis.
	'''
	
	reg = ['Periodicity', 'Syncope', 'Groove']
	b = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma1', 'Gamma2', 'HighGamma']
	
	if type(lobe) == list:
		title = "Average Amplitude per Channel for " + str(reg[r]) + " (" + str(lobe[0]) + ", " + str(lobe[1]) + ")"
	else: 
		title = "Average Amplitude per Channel for " + str(reg[r]) + " (" + str(lobe) + ")"

	m0 = np.mean(X_c0[:,:], axis = 0)
	std0 = np.std(X_c0[:,:], axis = 0)
	m1 = np.mean(X_c1[:,:], axis = 0)
	std1 = np.std(X_c1[:,:], axis = 0)
	m2 = np.mean(X_c2[:,:], axis = 0)
	std2 = np.std(X_c2[:,:], axis = 0)

	d0 = {'Mean' : m0, 'std' : std0, 'Perception' : 'Low'}
	dd0 = pd.DataFrame(data = d0)
	d1 = {'Mean' : m1, 'std' : std1, 'Perception' : 'Medium'}
	dd1 = pd.DataFrame(data = d1)
	d2 = {'Mean' : m2, 'std' : std2, 'Perception' : 'High'}
	dd2 = pd.DataFrame(data = d2)

	dd0 = dd0.append(dd1) 	
	dd0 = dd0.append(dd2)

	fig = px.box(dd0, x = "Perception", y="Mean", points="all", color = 'Perception', 
			  title = title, width=1200, height=800)
	#fig.write_html('desc_X.html', auto_open=True)
	if moy == True:
		titlesvg = "boxplot_" + str(reg[r] + "_p" + str(isuj)) + "_" + str(b[f]) + ".svg"
		titlepng = "boxplot_" + str(reg[r] + "_p" + str(isuj)) + "_" + str(b[f]) + ".png"

	else:
		titlesvg = "boxplot_" + str(reg[r] + "_p" + str(isuj)) + "_" + str(b[f]) + ".svg"
		titlepng = "boxplot_" + str(reg[r] + "_p" + str(isuj)) + "_" + str(b[f]) + ".png"

	fig.write_image(str(titlesvg))
	fig.write_image(str(titlepng))
	fig.show()
	 

	
def trans_y(y, r):
	'''
	Parameters
	----------
	y : target array.
	r : regressor.

	Returns
	-------
	y : target array with the good category.

	'''
	# Groove
	if r == 2 :
		y = y.reshape(len(y), 1)
		y = pd.DataFrame(y) ; y = round(y, 6) ; y = y.astype(str)
		l = np.unique(y) ; y = y.replace(list(l), list(range(0,len(l))))
		
		if np.unique(y)[-1] == 3:
			y = y.replace([0, 1], [0, 0]) ; y = y.replace([2], [1]) ; y = y.replace([3, 4, 5], [2, 2, 2])
		
		else: 
			print("   /!\ Unbalanced transformation")
			y = y.replace([0, 1], [0, 0]) ; y = y.replace([2, 3], [1, 1]) ; y = y.replace([4, 5], [2, 2])
		y = np.array(y.astype(float))
	
	# Syncope
	if r == 1:
		y = y.reshape(len(y), 1)
		y = pd.DataFrame(y) ; y = round(y, 6) ; y = y.astype(str)
		l = np.unique(y) ; y = y.replace(list(l), list(range(0,len(l))))
		y = y.replace([0, 1, 2, 3], [0, 0, 0, 0]) ; y = y.replace([4, 5, 6, 7], [1, 1, 1, 1]) ; y = y.replace([8, 9, 10, 11,12], [2, 2, 2, 2, 2])
		y = np.array(y.astype(float))
	
	# Periodicity
	if r == 0: 
		y = y.reshape(len(y), 1)
		y = pd.DataFrame(y) ; y = round(y, 6) ; y = y.astype(str)
		l = np.unique(y) ; y = y.replace(list(l), list(range(0,len(l))))
		y = y.replace([0], [0]) ; y = y.replace([1], [1]) 
		y = np.array(y.astype(float))
		
	return y 


def plot_gr_sy(groove, syncope):
	fig = px.scatter(x=groove, y=syncope, trendline="ols")
	fig.write_html('desc_syncgroove.html', auto_open=True)
	fig.show()



def summary_stat(isuj, r, lobe, moy, f, nlobe, plt, extractGC, binary, bande):
	''' Parameters
	----------
	isuj : subject (int).
	r : regressor (int).
	lobe : string.
	moy : if True return bands frequency (bool).
	f : 0-7 if moy = True, 0-99 else (int).
	nlobe : number of lobe (0-4).
	plt : if True plot descriptives analysis.
	extractGC : if True return the goodchan (var criterion).
	binary : if True return only low and high (bool).

	Returns
	-------
	X : Array data with nlobe.
	y : Array target.
	dictChann : association channels idx theoricals and reals.
	idx : idx size.

	'''
	
	if nlobe == 1: 
		X, y, dictChann, lengc, idx = subsample(isuj = isuj, r = r, lobe = lobe, moy = moy, f = f[0], lengc = 0, bande = bande)
		
	if nlobe == 2: 
		X, y, dictChann, lengc, _ = subsample(isuj = isuj, r = r, lobe = lobe[0], moy = moy, f = f[0], lengc = 0, bande = bande)
		X1, y1, dictChann1, _ , idx = subsample(isuj = isuj, r = r, lobe = lobe[1], moy = moy, f = f[1], lengc = lengc,bande = bande)
		X2 = np.concatenate((X, X1), axis = 1)
		dictChann.update(dictChann1)
		X = X2
		
	if nlobe == 3:
		idx = []
		X, y, dictChann, lengc, idx1 = subsample(isuj = isuj, r = r, lobe = lobe[0], moy = moy, f = f[0], lengc = 0, bande = bande)
		X1, y1, dictChann1,lengc1 , idx2 = subsample(isuj = isuj, r = r, lobe = lobe[1], moy = moy, f = f[1], lengc = lengc, bande = bande)
		X2, y2, dictChann2, _ , idx3= subsample(isuj = isuj, r = r, lobe = lobe[2], moy = moy, f = f[2], lengc = (lengc + lengc1), bande = bande)
		X3 = np.concatenate((X, X1), axis = 1)
		X4 = np.concatenate((X2, X3), axis = 1)
		dictChann.update(dictChann1)
		dictChann.update(dictChann2)
		idx.append(idx1) ; 	idx.append(idx2) ; 	idx.append(idx3)
		X = X4
	
	if nlobe == 4:
		idx = []
		X, y, dictChann, lengc, idx1 = subsample(isuj = isuj, r = r, lobe = lobe[0], moy = moy, f = f[0], lengc = 0, bande = bande)
		X1, y1, dictChann1, lengc1 , idx2 = subsample(isuj = isuj, r = r, lobe = lobe[1], moy = moy, f = f[1], lengc = lengc, bande = bande)
		X2, y2, dictChann2, lengc2 , idx3 = subsample(isuj = isuj, r = r, lobe = lobe[2], moy = moy, f = f[2], lengc = (lengc + lengc1), bande = bande)
		X3, y3, dictChann3, _ , idx4 = subsample(isuj = isuj, r = r, lobe = lobe[3], moy = moy, f = f[3], lengc =(lengc + lengc1+lengc2), bande = bande)
		X4 = np.concatenate((X, X1), axis = 1)
		X5 = np.concatenate((X2, X3), axis = 1)
		X6 = np.concatenate((X4, X5), axis = 1)
		dictChann.update(dictChann1)
		dictChann.update(dictChann2)
		dictChann.update(dictChann3)
		idx.append(idx1) ; 	idx.append(idx2) ; 	idx.append(idx3) ; idx.append(idx4)
		X = X6
		
	y = trans_y(y = y, r = r)
	df_count, dict_idx = desc_y(y, r)
	X = StandardScaler().fit_transform(X)
	
	if r != 0:  
		X_c0, X_c1, X_c2 = desc_X(X, dict_idx, r)
		
		if plt == True:
			plot_desc_y(df_count, len(y), r, isuj)
			plot_desc_X(X_c0, X_c1, X_c2, r, lobe, isuj, f, moy)
			
		if extractGC == True: 
			# Retourne les sous-échantillons précédent moyennés 
			dd0, dd1, dd2 = samplecatX(X_c0, X_c1, X_c2)
			
			# Retourne les indices des canaux n ayant enlevé les 10 canaux les plus variables
			idxGoodChan = extract_badChann(dd0, dd1, dd2)
			X = X[:, idxGoodChan]
		
		if binary == True:
			X = np.concatenate((X_c0, X_c2), axis = 0)
			y = y[np.where((y == 0) | (y == 2))]
			
			return X, y, dictChann, idx
		
		else:
			return X, y, dictChann, idx
	
	else: 
		if plt == True: 
			plot_desc_y(df_count, len(y), r, isuj)
		X_c0, X_c1 = desc_X(X, dict_idx, r)
		return X, y, dictChann, idx
		
		
def summary_all_lobes(isuj, r, lobe, moy, f, extractGC = True, bande = 'classic'):
	'''
	Parameters
	----------
	isuj : subject number (int).
	r : regressor 0,1,2 (int).
	lobe : lobe's name (string).
	moy : True if average of each frequency band (bool).
	f : frequency (0-6 included if moy = True, else 0<f<99) (int)
	extractGC : BOOL, optional
		If true put out the bad chann (variance criterion). The default is True.

	Returns
	-------
	dict_summay : dictionnary which summaries X informations.

	'''
	X, y, dictChann, lengc, idx = subsample(isuj = isuj, r = r, lobe = lobe, moy = moy, f = f, lengc = 0, bande = bande)
	
	if r != 0: 
		y = trans_y(y = y, r = r)
		df_count, dict_idx = desc_y(y, r)
		X = StandardScaler().fit_transform(X)
		#X = MinMaxScaler().fit_transform(X)
		X_c0, X_c1, X_c2 = desc_X(X, dict_idx, r)
		
		if extractGC == True: 
			# Retourne les sous-échantillons précédent moyennés 
			dd0, dd1, dd2 = samplecatX(X_c0, X_c1, X_c2)
			
			# Retourne les indices des canauxe n ayant enlevé les 10 canaux les plus variables
			idxGoodChan = extract_badChann(dd0, dd1, dd2)
			X_c0 = X_c0[:, idxGoodChan]
			X_c1 = X_c1[:, idxGoodChan]
			X_c2 = X_c2[:, idxGoodChan]
			
		dict_summay = {
			"X_c0" : X_c0,
			"X_c1" : X_c1, 
			"X_c2" : X_c2, 
			"r" : r,
			"lobe" : lobe, 
			"f" : f}
	else:
		y = trans_y(y = y, r = r)
		df_count, dict_idx = desc_y(y, r)
		X = StandardScaler().fit_transform(X)
		# X = MinMaxScaler().fit_transform(X)
		X_c0, X_c1 = desc_X(X, dict_idx, r)
			
		dict_summay = {
			"X_c0" : X_c0,
			"X_c1" : X_c1, 
			"r" : r,
			"lobe" : lobe, 
			"f" : f}
	
	return dict_summay


def plot_lobes(lds, p, r, plot, bande = 'classic'):
	'''
	Parameters
	----------
	lds : list of lobes innformation.
	p : participants.
	r : regressor.
	plot : ploting condition.

	Returns
	-------
	Show and save plot. '''
	
	reg = ['Periodicity', 'Syncope', 'Groove']
	
	b = ['Delta', 'Theta', 'Alpha', 'Beta Low','Beta High', 'Gamma1', 'Gamma2', 'HighGamma']
	
	associatelobes = {
		'LFidx' : 'Frontal',
		'LTGidx' : 'Temporal Left',
		'LTDidx' : 'Temporal Right',
		'LPidx' : 'Parietal',
		'LOidx' : 'Occipital'}
	
	finaldf = pd.DataFrame()
	
	ds1 = lds[0]
	title = "Average Amplitude per Channel for the " + str(b[ds1['f']]) + " Band (Regressor: " + str(reg[r]) + ", subject: " + str(p) + ")"
	
	for ds in lds:
		
		m0 = np.mean(ds['X_c0'][:,:], axis = 0) ; std0 = np.std(ds['X_c0'][:,:], axis = 0)
		m1 = np.mean(ds['X_c1'][:,:], axis = 0) ; std1 = np.std(ds['X_c1'][:,:], axis = 0)
		
		if r != 0: 
			m2 = np.mean(ds['X_c2'][:,:], axis = 0) ; std2 = np.std(ds['X_c2'][:,:], axis = 0)
	
		d0 = {'Mean' : m0, 'std' : std0, 'Perception' : 'Low', 'Lobe' : associatelobes[ds['lobe']], 'freq': b[ds['f']]}
		dd0 = pd.DataFrame(data = d0)
		d1 = {'Mean' : m1, 'std' : std1, 'Perception' : 'Medium', 'Lobe' : associatelobes[ds['lobe']],'freq': b[ds['f']]}
		dd1 = pd.DataFrame(data = d1)
		
		if r != 0:
			d2 = {'Mean' : m2, 'std' : std2, 'Perception' : 'High', 'Lobe':associatelobes[ds['lobe']], 'freq': b[ds['f']]}
			dd2 = pd.DataFrame(data = d2)
	
		dd0 = dd0.append(dd1) 
		
		if r != 0: 
			dd0 = dd0.append(dd2)
			
		finaldf = finaldf.append(dd0)
	
	if plot == 'bycond':
		fig = px.box(finaldf, x = "Perception", y="Mean", points="all", color = 'Lobe', 
				  title = title, width=2000, height=1200)
		fig.update_layout(
		    title=title,
		    yaxis_range=[-0.5,0.6],
			font_family="Times New Roman",
			font_size=25)
		#fig.write_html('desc_X.html', auto_open=True)

		titlesvg = "boxplot_bycond_" + str(reg[r] + "_p" + str(isuj)) + "_f" + str(b[f]) + ".svg"
		titlepng = "boxplot_" + str(reg[r] + "_p" + str(isuj)) + "_f" + str(b[f]) + "_v2.png"
		fig.write_image(str(titlesvg))
		fig.write_image(str(titlepng))
		fig.show()
	
	if plot == 'bylobe':
		fig = px.box(finaldf, x = "freq", y="Mean", points="all", color = 'Perception', 
				  title = title, width=2000, height=1200)
		fig.update_layout(
		    title=title,
		    yaxis_range=[-0.5,0.6],
			font_family="Times New Roman",
			font_size=25)
		#fig.write_html('desc_X.html', auto_open=True)

		titlesvg = "boxplot_bylobe" + str(reg[r] + "_p" + str(isuj)) + "_f" + str(b[ds1['f']]) + "_" + str(associatelobes[ds1["lobe"]]) + ".svg"
		titlepng = "boxplot_bylobe" + str(reg[r] + "_p" + str(isuj)) + "_f" + str(b[ds1['f']]) + "_" + str(associatelobes[ds1["lobe"]]) + ".png"
		fig.write_image(str(titlesvg))
		fig.write_image(str(titlepng))
		fig.show()
	
	return finaldf
	
################################################################################
# Utilities functions

def merge_dict(chanT,chanR): 
	# Return idx 0-len(dict) for theorical and real idx
	ladd= list(range(0, len(chanR)))
	idxT = dict(zip(chanT['name'], ladd))
	idxR = dict(zip(chanR['Unnamed: 0'], ladd))
	return idxR, idxT
	
def tradix(idxR, idxT):
	# Return idxRT which is the dict association of theorical idx and reals
	liR = [] ; liT = []
	for key, value in idxT.items():
		print(key)
		iR = idxR[key]
		iT = value
		liR.append(iR)
		liT.append(iT)
	idxRT = dict(zip(liT, liR))
	return idxRT
	
def get_key(val, dictItem):
	# Retourne la clé associée à la valeur
	for key, value in dictItem.items():
		if val == value:
			print("val : ", val, "key", key)
			return key	
	
def make_dict(lidxTh):
	# Cette foncion retourne l'indice channel associé à la l'index localisation théorique 
	ladd = list(range(0, len(lidxTh)))
	dictMerge= dict(zip(ladd, lidxTh))
	return dictMerge
	
def idx_change_toReal(lidx, idxRT):
	# ??? 
	idxChanged = []
	for i in lidx:
		val = get_key(i, idxRT)
		idxChanged.append(val)
	return idxChanged

################################################################################


def compareReg(isuj, r = [0, 1, 2], lobe = "LFidx", moy = True, f = [0], lengc = 0):
	'''
	Parameters
	----------
	isuj : subject number
	r : regressor list. The default is [0, 1, 2].
	lobe : no importance to compare regressor. The default is "LFidx".
	moy : Frequency bannd or not. The default is True.
	f :  The default is [0].
	lengc : TYPE, optional
		DESCRIPTION. The default is 0.

	Returns
	-------
	plot  response between syncope groove and periodicity.'''
	
	X, target, dictionary, _, _ = subsample(isuj, r = r, lobe = lobe, moy = moy, f = f[0], lengc = 0)
	p = trans_y(y = target[:, 0], r = 0)
	p = p.reshape(len(p))
	s = trans_y(y = target[:, 1], r = 1)
	s = s.reshape(len(s))
	g = trans_y(y = target[:, 2], r = 2)
	g = g.reshape(len(g))
	
	dd = {
	   'periodicity' : p,
	   'syncope' : s,
	   'groove' : g} 

	ddd = pd.DataFrame(data = dd)
	
	p0 = ddd[(ddd.periodicity == 0)] ; p1 = ddd[(ddd.periodicity == 1)]
	y0s, _ = desc_y(np.array(p0.syncope),1) ; y1s, _ = desc_y(np.array(p1.syncope),1)
	y0g, _ = desc_y(np.array(p0.groove),2) ; y1g, _ = desc_y(np.array(p1.groove),2)
	
	ys = y0s.append(y1s) ; ys = ys.reset_index() ; ys['index'][1] = 1
	yg = y0g.append(y1g) ; yg = yg.reset_index(); yg['index'][1] = 1
	
	title = "Distribution of classes for Groove"
	l = len(ddd)
	y = yg
	fig = px.bar(y, x='index', y=['Low', 'Medium', 'High'], 
			   title = title, height=800, width=1200)
	fig.add_hline(y=l/3)
	fig.add_hline(y=l/3 + 15, line_dash="dash")
	fig.add_hline(y=l/3 - 15, line_dash="dash")
	
	titlesvg = "barplotGroove"  + "_p" + str(isuj) + "_.svg"
	titlepng= "barplotGroove" +  "_p" + str(isuj) + "_.png"
	fig.write_image(str(titlesvg))
	fig.write_image(str(titlepng))
	fig.show()
	
	title = "Distribution of classes for Syncope"
	l = len(ddd)
	y = ys
	fig = px.bar(y, x='index', y=['Low', 'Medium', 'High'], 
			   title = title, height=800, width=1200)
	fig.add_hline(y=l/3)
	fig.add_hline(y=l/3 + 15, line_dash="dash")
	fig.add_hline(y=l/3 - 15, line_dash="dash")
	
	titlesvg = "barplotSyncope"  + "_p" + str(isuj) + "_.svg"
	titlepng= "barplotSyncope" +  "_p" + str(isuj) + "_.png"
	fig.write_image(str(titlesvg))
	fig.write_image(str(titlepng))
	fig.show()
	
	
	
if __name__ == '__main__':
	
	
	# OSError: [Errno 28] No space left on device: 'boxplot_bylobeGroove_p1_fDelta_Temporal Right.svg'
	
	path = '/Users/swann/Université/Stage/Spyder/Analysis.'
	path_sl = 'sort_loc.csv'
	data_chanReal = pd.read_csv(path_sl)
	path = '4D248.csv'
	data_chanTh = pd.read_csv(path, sep=';')
	data_chanTh = data_chanTh[0:248]
	
	# 1. BoxPlot pour tous les lobes pour chaque fréquence 
	run_lobe_freq = False
	
	# 2. BoxPlot pour toutes les fréquences pour chaque lobe
	run_freq_1lobe = True
	
	# 3. BarPlot pour les régresseurs (répartition syncope et groove en fonction de la périodicité)
	run_compareReg = False
	
	# 4. Barplot pour chaque régresseur
	run_DescYX = False
	
	t_test = True
	
	
	# 1
	if run_lobe_freq == True : 
		
		isuj = 1
		r = 2
		freq = [6]
		nlobe = 3
		moy = True
		lobe = ["LTDidx", "LPidx", "LFidx", 'LTGidx', 'LOidx']
		bisuj =  [1]
		#bisuj = [0, 7, 21, 23, 1]
		# bisuj = [ 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29]
		bande = 'classic'
	
		for f in freq:
			for isuj in bisuj:
				d1 = summary_all_lobes(isuj, r, "LTDidx", moy, f, extractGC = True, bande = bande)
				d2 = summary_all_lobes(isuj, r, "LTGidx", moy, f, extractGC = True, bande = bande)
				d3 = summary_all_lobes(isuj, r, "LFidx", moy, f, extractGC = True, bande = bande)
				d4 = summary_all_lobes(isuj, r, "LPidx", moy, f, extractGC = True, bande = bande)
				d5 = summary_all_lobes(isuj, r, "LOidx", moy, f, extractGC = True, bande = bande)
				listds = [d1, d2, d3, d4, d5]
				df = plot_lobes(listds, isuj, r, plot = "bycond")
	
	
	# 2
	if run_freq_1lobe == True:
		
		bande = 'classic'
		# lobe = ["LTDidx", "LPidx", "LFidx", "LTGidx", "LOidx"]
		lobe = ["LFidx"] # 
		bisuj =  [1]
		r = 2
		moy = True
		nlobe = 1
	
		for isuj in bisuj:
			for l in lobe:
				d1 = summary_all_lobes(isuj, r, l, moy, 0, extractGC = False, bande = bande) #Delta
				d2 = summary_all_lobes(isuj, r, l, moy, 1, extractGC = False, bande = bande) #Theta
				d3 = summary_all_lobes(isuj, r, l, moy, 2, extractGC = False, bande = bande) #Alpha
				d4 = summary_all_lobes(isuj, r, l, moy, 3, extractGC = False, bande = bande) #Beta1
				d5 = summary_all_lobes(isuj, r, l, moy, 4, extractGC = False, bande = bande) #Beta2
				d6 = summary_all_lobes(isuj, r, l, moy, 5, extractGC = False, bande = bande) #Low 1 gamma
				d7 = summary_all_lobes(isuj, r, l, moy, 6, extractGC = False, bande = bande) # Low 2 gamma
				d8 = summary_all_lobes(isuj, r, l, moy, 7, extractGC = False, bande = bande) # high gamma
				listds = [d1, d2, d3, d4, d5, d6, d7, d8]
				df = plot_lobes(listds, isuj, r, plot = "bylobe")
		

	# 3	
	if run_compareReg == True: 
		
		bisuj =  [0, 1, 7, 21, 23]
		
		for isuj in bisuj:
			compareReg(isuj)
			# X, y, dictChann, idx  = summary_stat(isuj, r, lobe, moy, f, nlobe, True, False, False)


	# 4
	if run_DescYX == True:

		isuj = 0 
		r = 1
		f = 0
		
		bande = 'classic'
		lobe = "LTDidx"
		nlobe = 1
		
		moy = True
		binary = False
		plt = True
		
		X,y, idx_LobeRchan, lidx = summary_stat(isuj, r, lobe, moy, f, nlobe, plt, False, binary, bande = bande)
		df_count, dict_idx = desc_y(y)
		
		
	if t_test == True:
		
		'''
		isuj = 1
		r = 0
		freq = [2]
		nlobe = 3
		moy = False
		lobe = ["LTDidx", "LPidx", "LFidx", 'LTGidx', 'LOidx']
		bisuj =  [1]
		# bisuj = [ 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29]
		bande = 'classic'
		
		for f in freq:
			for isuj in bisuj:
				d1 = summary_all_lobes(isuj, r, "LTDidx", moy, f, extractGC = False, bande = bande)
				d2 = summary_all_lobes(isuj, r, "LFidx", moy, f, extractGC = False, bande = bande)
				d3 = summary_all_lobes(isuj, r, "LPidx", moy, f, extractGC = False, bande = bande)
		'''
	
		from scipy.stats import shapiro
		import scipy.stats as stats
		from scipy.stats import norm
		
		listd = [d1, d2, d3, d4, d5, d6, d7, d8]
		lobes = ['Temporal Right', 'Temporal Left', 'Frontal', 'Parietal', 'Occipital' ]
		freq = ['Delta', 'Theta', 'Alpha', 'Low Beta', 'High Beta', 'Low 1 Gamma', 'Low 2 Gamma', "High Gamma"]
		resume = {
			'shap0': [],
			'shap1':[],
			'shap2':[],
			'pval0':[],
			'pval1':[],
			'pval2':[],
			'val_stud_low_med' : [],
			'val_stud_low_high' : [],
			'val_stud_med_high' : [],

			'pval_stud_low_med': [],
			'pval_stud_low_high': [],
			'pval_stud_med_high': [],

			'f': [] # chaneg by lobe
			} 
		
		i=0
		f = 0
		for d in listd:
			m0 = np.mean(d['X_c0'][:,:], axis = 0) 
			m1 = np.mean(d['X_c1'][:,:], axis = 0) 
			m2 = np.mean(d['X_c2'][:,:], axis = 0) 

			x1, pval1 = shapiro(m0)
			x2, pval2 = shapiro(m1)
			x3, pval3 = shapiro(m2)


			resume['shap0'].append(round(x1, 3))
			resume['shap1'].append(round(x2, 3))
			resume['shap2'].append(round(x3, 3))

			resume['pval0'].append(round(pval1, 3))
			resume['pval1'].append(round(pval2, 3))
			resume['pval2'].append(round(pval3, 3))
			resume['f'].append(f)
			
			x = np.linspace(-0.3, 0.35)

			plt.hist(m0, color = 'royalblue', edgecolor = 'black',
				  bins = int(len(m0)), label = "Low")
			mu, std = norm.fit(m0)
			plt.plot(x, stats.norm.pdf(x, mu, std), color = 'royalblue')
			
			plt.hist(m1, color = 'indianred', edgecolor = 'black',
				  bins = int(len(m0)), label = "Medium")
			mu, std = norm.fit(m1)
			plt.plot(x, stats.norm.pdf(x, mu, std), color = 'indianred')
			
			plt.hist(m2, color = 'mediumaquamarine', edgecolor = 'black',
				  bins = int(len(m0)), label = "High")
			mu, std = norm.fit(m2)
			plt.plot(x, stats.norm.pdf(x, mu, std), color = 'mediumaquamarine')
			
			plt.title("Frequency Band:" + str(freq[f]))
			plt.legend()
			title = "p1_groove_F_f" + str(f) + ".svg"
			plt.savefig(str(title), format="svg")
			plt.show()
			
			f0, fpvalue0 = stats.ttest_ind(m0,m1, equal_var = False)
			f1, fpvalue1 = stats.ttest_ind(m0,m2, equal_var = False)
			f2, fpvalue2 = stats.ttest_ind(m1,m2, equal_var = False)

			resume['val_stud_low_med'].append(round(f0,3))
			resume['val_stud_low_high'].append(round(f1,3))
			resume['val_stud_med_high'].append(round(f2,3))
			resume['pval_stud_low_med'].append(round(fpvalue0,3))
			resume['pval_stud_low_high'].append(round(fpvalue1,3))
			resume['pval_stud_med_high'].append(round(fpvalue2,3))
			
			f = f +1
			
		resume_df = pd.DataFrame(data = resume)
		
		import csv
		resume_df.to_csv("resume_df_groove_F.csv")
		print(resume_df)
			
		
	# Retourne sous échantillon (lobe) de X, y avec dict des indices dans les données réelles 
	# listisuj = [0, 1, 4, 6, 7, 9, 10, 12, 13, 14, 18, 20, 21, 22, 23, 25, 26, 27]
	# for isuj in listisuj:
		# X,y, idx_LobeRchan, lidx = summary_stat(isuj, r, lobe, moy, f, nlobe, True)
	
	# X, y, dictChann, idx = summary_stat(isuj, r, lobe, moy, f, nlobe, False, False)
	
	# retourne deux dictionnaires d'indices associant le nom du canal à 
	# idxr = indices réels, et idxt indices théoriques
	# idxR, idxT = merge_dict(data_chanTh, data_chanReal)
	
	# Retourne l'index théorique (value) selon l'index réel (key)
	# idxRT = tradix(idxR, idxT)
	
	'''
	# Retourne les indices des essais selon la classes 
	df_count, dict_idx = desc_y(y)
	
	# Retourne les sous-échantillons X par classe
	X = StandardScaler().fit_transform(X)
	X_c0, X_c1, X_c2 = desc_X(X, dict_idx)
	
	# Retourne les sous-échantillons précédent moyennés 
	dd0, dd1, dd2 = samplecatX(X_c0, X_c1, X_c2)
	
	# Transforme l'index réel en index théorique 
	lidxTh = idx_change_toReal(lidx, idxRT)
	
	# En gros la clé correspond à l'index des données extraires d'un lobe, et la valeur associée est l'index pour la spatialisation
	lidxColor = make_dict(lidxTh)
	
	# 
	col0, col1, col2 = plotcol2(dd0, dd1, dd2, lidxTh, lidxColor)
	plot_2D(col0, data_chanTh, isuj, r, lobe, f, "low")
	plot_2D(col1, data_chanTh, isuj, r, lobe, f, "medium")
	plot_2D(col2, data_chanTh, isuj, r, lobe, f, "high")
	'''

	
	'''
	i = 1
	X, y, dictChann, lengc = subsample(isuj = 0, r = 2, lobe = 'LFidx', moy = True, f = 6, lengc = 0)
	X1, y1, dictChann1, _ = subsample(isuj = 0, r = 2, lobe = 'LTGidx', moy = False, f = 0, lengc = lengc)
	X2 = np.concatenate((X, X1), axis = 1)
	dictChann.update(dictChann1)
	
	
	y = trans_y(y = y, r = 2)
	df_count, dict_idx = desc_y(y)
	X = StandardScaler().fit_transform(X)
	X_c0, X_c1, X_c2 = desc_X(X, dict_idx)
	
	plot_desc_y(df_count, len(y), 2)
	plot_desc_X(X_c0, X_c1, X_c2, 2)
	'''
	'''
	X, y, dictChann, lengc = subsample(isuj = i, r = 2, lobe = 'LFidx', moy = True, f = 6, lengc = 0)
	yg = trans_y(y=y, r=2)
	yg = yg.flatten().tolist()
	X, y, dictChann, lengc = subsample(isuj = i, r = 1, lobe = 'LFidx', moy = True, f = 6, lengc = 0)
	ys = trans_y(y=y, r=1)
	ys = ys.flatten().tolist()
	plot_gr_sy(yg, ys)
	'''
