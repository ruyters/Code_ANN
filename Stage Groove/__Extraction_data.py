
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:19:30 2022

@author: Swann
"""

import scipy.io as sio
import numpy as np
# from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
# 													cross_val_multiscore, LinearModel, get_coef)

import pandas as pd




def extraction_data(raw, suj, moy = False):
		'''
		Parameters
		----------
		raw : matlab file who contains try and target.
		suj : number of subject 
		moy : extract by type of cerebral activity
				DESCRIPTION. The default is False.
				
				DELTA (0,1-4Hz), THETA (4-8Hz), ALPHA (8-12Hz), 
				BETA (12 - 40Hz), (peut varier selon la littérature considérée)
				GAMMA (40-80Hz), HAUT GAMMA (80-100Hz)
				

		Returns : TUPLE
		-------
		SUJ : multidimensionnal array.
		REG : target. '''
		
		# Data
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

		bandes = [0, 5, 9, 13, 41, 61, 81, 100]
		
		# Shape of output
		if moy == False:
			freq = 100
		else : 
			freq = len(bandes)
			
		data_trans = np.empty([30, 144, 246, freq])
		data_trans2 = data_trans.copy()
		target = np.empty([30, 144, 3])
		good_chan_idx = []
		
		# Subjects
		for isuj in range(len(suj)): 
				
				# print("Subject", isuj)
				SUJ = raw['Y'][isuj] # MEG
				REG = raw['X'][isuj,:,0:3] # 3 regressors of interest: periodicity, syncope, groove
				GoodChannel = GoodChannel_ImagingKernel['GoodChannel'][:,isuj]
				
				# remove bad trials
				goodtrial = np.reshape(bad2['badtrial_bnprocess'][isuj], 144)
				goodtrial = np.where(goodtrial == 1)[0]
				SUJ = SUJ[goodtrial,:,:]; REG = REG[goodtrial,:] # include/exclude bads
				
				# remove bad channels
				goodchan = np.zeros((299))
				goodchan[np.concatenate(GoodChannel[0])] = 1
				goodchan = goodchan[bad1['iMeg_allsuj'][0]]
				goodchan = np.where(goodchan == 1)[0]
				SUJ = SUJ[:,goodchan,:]
				
				
				if moy == False: # False : return final array with all frequencies
				# Final normal dataset
						tab = np.zeros([len(goodtrial), len(goodchan), 100])
						tab = SUJ.copy()
					
						# Reshape data to output shape
						a = 144 - tab.shape[0]; b = 246 - tab.shape[1] 
						good_chan_idx.append([isuj,tab.shape[0],tab.shape[1]])
						
						
						if (a > 0):
								tab_to_add = np.empty([a, tab.shape[1], 100])
								tab_to_add[:,:,:] = np.NaN
								tab_to_add = np.nan_to_num(tab_to_add, nan = 0)
								tab_reshape = np.append(tab, tab_to_add)
								tab = tab_reshape.reshape(144, tab.shape[1], 100)
 
						if (b > 0): 
								tab_to_add = np.empty([144, b, 100])
								tab_to_add[:,:,:] = np.NaN
								tab_to_add = np.nan_to_num(tab_to_add, nan = 0)
								tab_reshape = np.append(tab, tab_to_add)
								tab = tab_reshape.reshape(144, 246, 100)
						
						data_trans2[isuj, :, :, :] = tab[:,:,:]

				# Frequency transformation
				if moy == True: # True : return dynamically arrays with 6 frequencies
						data = SUJ.copy()
						bandes = [0, 4, 9, 13, 41, 61, 81, 100]
						tab = np.zeros([len(goodtrial), len(goodchan), len(bandes)])
						good_chan_idx.append([isuj,tab.shape[0],tab.shape[1]])
						
						for i in range(1,len(bandes)) :
								tab[:,:, i-1] = np.mean(data[:,:, bandes[i-1] : bandes[i]], axis=2)
						
					
						# Reshape data to output shape
						a = 144 - tab.shape[0]; b = 246 - tab.shape[1]  
						if (a > 0):
								tab_to_add = np.empty([a, tab.shape[1], len(bandes)])
								tab_to_add[:,:,:] = np.NaN
								tab_to_add = np.nan_to_num(tab_to_add, nan = 0)
								tab_reshape = np.append(tab, tab_to_add)
								tab = tab_reshape.reshape(144, tab.shape[1], len(bandes))
 
						if (b > 0): 
								tab_to_add = np.empty([144, b, len(bandes)])
								tab_to_add[:,:,:] = np.NaN
								tab_to_add = np.nan_to_num(tab_to_add, nan = 0)
								tab_reshape = np.append(tab, tab_to_add)
								tab = tab_reshape.reshape(144, 246, len(bandes))
								
						# Add to final ndarray
						data_trans2[isuj, :, :, :] = tab[:,:,:]


				# Reshape target
				a = 144 - REG.shape[0]
				tab_to_add = np.empty([a, 3])
				tab_to_add[:,:] = np.NaN
				tab_to_add = np.nan_to_num(tab_to_add, nan = 0)

				tab_reshape = np.append(REG, tab_to_add)
				REG = tab_reshape.reshape(144, 3)
				target[isuj,:, :] = REG[:,:]
		
		print("Extraction done")
		return data_trans2, target, good_chan_idx
		
				
			 
def extract_dyn(isuj):
		 
		# Data
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

		SUJ = raw['Y'][isuj] # MEG
		REG = raw['X'][isuj,:,0:3] # 3 regressors of interest: periodicity, syncope, groove
		GoodChannel = GoodChannel_ImagingKernel['GoodChannel'][:,isuj]
		
		# remove bad trials
		goodtrial = np.reshape(bad2['badtrial_bnprocess'][isuj], 144)
		goodtrial = np.where(goodtrial == 1)[0]
		SUJ = SUJ[goodtrial,:,:]; REG = REG[goodtrial,:] # include/exclude bads
		
		# remove bad channels
		goodchan = np.zeros((299))
		goodchan[np.concatenate(GoodChannel[0])] = 1
		goodchan = goodchan[bad1['iMeg_allsuj'][isuj]]
		goodchan = np.where(goodchan == 1)[0]
		SUJ = SUJ[:,goodchan,:]
		
		# Mean frequencies
		data = SUJ.copy()
		bandes = [0, 4, 9, 13, 41, 51, 61, 71, 91, 100]
		tab = np.zeros([len(goodtrial), len(goodchan), len(bandes)])
		
		for i in range(1,len(bandes)) :
				tab[:,:, i-1] = np.mean(data[:,:, bandes[i-1] : bandes[i]], axis=2)
		
		return tab, REG
				




