# -*- coding: utf-8 -*-
import csv
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as sio
import plotly.graph_objects as go
import numpy as np
import scipy
import joblib

from __ExtractDyn import *


'''
Les fonctions ci-dessous permettent d'obtenir les visualisations par rapport aux coordonnées théoriques des canaux MEG
ainsi que de regarder soit les canaux sélectionnés par RFE avec le SVM et l'arbre de décision,
soit en fonction de leur importance (valeurs de shapley).
'''


def plotcol(LF, LP, LTG, LTD, LO):
	'''
	

	Parameters
	----------
	LF, LP, LTG, LTD, LO : capteurs MEG en fonction des lobes cérébraux

	Returns
	-------
	data_color : liste couleur correspondant aux lobes.

	'''
    data_color = []
    for i in range(1,249):
        if i in LF:
            data_color.append(0)
        if i in LP:
            data_color.append(1)
        if i in LTG : 
            data_color.append(2)
        if i in LTD:
            data_color.append(2)
        if i in LO:
            data_color.append(3)
    return data_color


def plotcol_RFE(idx):
	# Return the color list with RFE (col = goodchann)
    data_color = []
    for i in range(1,249):
        if i in idx:
            data_color.append(0)
        else:
            data_color.append(4)
    return data_color

def plotcol_RFE_DF(df):
	# Return the color list with RFE (col = goodchann)
    data_color = []
    for i in range(1,249):
        if i in df['channels_reals']:
            data_color.append(df['feature_importance_vals'][i])
        else:
            data_color.append(0)
    return data_color

		 
def plotcol2(dd0, dd1, dd2, lidxTh, lidxColor):
	
	data_color_d0 = []
	data_color_d1 =[]
	data_color_d2 = []
	
	for i in range(0, 249):
		if i in lidxTh:
			print(i, "append dO")
			valCol = get_key(i, lidxColor)
			if dd0['Mean'][valCol] < 0:
				data_color_d0.append(-1)		
			else:
				data_color_d0.append(1)
		else:
			print(i, "else")
			data_color_d0.append(0)
	
	for i in range(0, 248):
		if i in lidxTh:
			print(i, "append dO")
			valCol = get_key(i, lidxColor)
			if dd1['Mean'][valCol] < 0:
				data_color_d1.append(0)		
			else:
				data_color_d1.append(2)
		else:
			print(i, "else")
			data_color_d1.append(3)
	
	for i in range(0, 248):
		if i in lidxTh:
			print(i, "append dO")
			valCol = get_key(i, lidxColor)
			if dd2['Mean'][valCol] < 0:
				data_color_d2.append(0)		
			else:
				data_color_d2.append(2)
		else:
			print(i, "else")
			data_color_d2.append(3)
	
	return data_color_d0, data_color_d1, data_color_d2
			

def plot_2D(data_color, data_chan, isuj, r, lobe, f, mod = 'SVM'):
    reg = ['Periodicity', 'Syncope', 'Groove']
    b = ['Delta', 'Theta', 'Alpha', 'Beta', 'Beta1', 'Gamma1', 'Gamma2', 'HighGamma']
    title = "Average Activation for class " + str(mod) + " (subj: " + str(isuj) + ", lobe: " + str(lobe) + ", freq: ", b[f] + ")"
    x = data_chan['x']
    y = data_chan['y']
    fig = px.scatter(data_chan, x=x, y=y,
                     text=data_chan['name'], color=data_color, color_continuous_scale='Turbo',
					 size = data_color)
    #fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        #title = title,
        autosize=True,
        width=1000,
        height=1000)
    titlesvg = "mapActiv" + str(reg[r] + "_p" + str(isuj)) + "_" + str(b[f]) + ".svg"
    titlepng = "mapActiv" + str(reg[r] + "_p" + str(isuj)) + "_" + str(b[f]) + ".png"
    titlehtml = "mapAct" + str(mod) + ".html"
    fig.write_image(str(titlesvg))
    fig.write_image(str(titlepng))
    fig.write_html(str(titlehtml), auto_open=True)

def extract_idx(LF, LP, LTG, LTD, LO, data_chan):
    
    LPF = [] ;  LPidx = []
    LFF = []; LFidx = []
    LTGF = [] ; LTGidx = []
    LTDF = [] ; LTDidx = []
    LOF = [] ; LOidx = []
    
    for i in LF: 
        n = 'A' + str(i)
        LFF.append(n)
    for i in LP:
        n = 'A' + str(i)
        LPF.append(n)
    for i in LTG: 
        n = 'A' + str(i)
        LTGF.append(n)
    for i in LTD: 
        n = 'A' + str(i)
        LTDF.append(n)
    for i in LO: 
        n = 'A' + str(i)
        LOF.append(n)
    
    for i in range(0, 248):
        
        if data_chan['Unnamed: 0'][i] in LFF:
            LFidx.append(i)
        if data_chan['Unnamed: 0'][i] in LPF:
             LPidx.append(i)
        if data_chan['Unnamed: 0'][i] in LTGF:
             LTGidx.append(i)
        if data_chan['Unnamed: 0'][i] in LTDF:
             LTDidx.append(i)
        if data_chan['Unnamed: 0'][i] in LOF:
             LOidx.append(i)
    
    return LFidx, LPidx, LTGidx, LTDidx, LOidx

def idx_change(lidx, idxRT):
	# ??? 
	idxChanged = []
	for i in lidx:
		val = idxRT[i]
		idxChanged.append(val)
	return idxChanged


def idx_realtoth(data_chanReal):
	for i in range(0,248):
		data_chanReal['Unnamed: 0'][i] = data_chanReal['Unnamed: 0'][i][1:]
	
	return data_chanReal

if __name__ == "__main__":
	
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
	
	
	path = '/Users/swann/Université/Stage/Spyder/Analysis.'
	path_sl = 'sort_loc.csv'
	data_chanReal = pd.read_csv(path_sl)
	path = '4D248.csv'
	data_chanTh = pd.read_csv(path, sep=';')
	data_chanTh = data_chanTh[0:248]
	association_df = joblib.load('association_df.pkl')
	
	# Load Shapley features importance
	# df =  joblib.load('Shap_FI1ch_p1_t76_f[6, 3, 3].pkl')
	
	plot_Shapley = False
	
	if Plot_Shapley == True:
		# Récupérer les indicpes RFE
		# idx1 = joblib.load('idx_SVM_lobes1_allfreq.pkl')
		# idx2 = joblib.load('idx_SVM_lobes2_allfreq.pkl')
		
		# df1 = joblib.load('df_lobes1_allfreq.pkl')
		# idx_1l = idx1[30]
		# idx_2l = idx2[3]
		

		#dff_sort = dff.sort_values(by = ['feature_importance_vals'], ascending = False)
		df = df.sort_values(by=['feature_importance_vals'], ascending = False)[0:40]
		idx_1l = list((df['col_name']).astype(int))
		
		# Récupérer dictChann
		#Xlf, _, dictChann_lf, idx_lf = summary_stat(1, 2, 'LFidx', False, [2, 2, 2], 1, False, False, True, 'classic')
		# Xlpltd, _, dictChann_lpltd, idx_lpltd = summary_stat(1, 2, ['LPidx', 'LTDidx'], False, [2, 2, 2], 2, False, False, True, 'classic')
		# X, _, dictChann, idx = summary_stat(1, 2, ['LPidx', 'LFidx', 'LTDidx', 'LTGidx'], False, [2, 2, 2, 2], 3, False, False, True, 'classic')

		#X = X[:, idx_1l]

		'''
		Xlf = Xlf[:, idx_1l]
		Xlpltd = Xlpltd[:, idx_2l
		X = np.append(Xlf, Xlpltd, axis = 1)
		X = joblib.dump(X, "X.pkl")
		'''
	
		# Lien entre dictChann théorique et mes capteurs RFE
		dc = np.asarray(list(dictChann.values()))
		dc = dc[idx_1l]
		
	
		dChan = dictChann.copy()
		for key in dictChann:
			if key in idx_1l:
				pass
			else:
				del dChan[key]

		dSummary = {
			'idx_RFE' : dChan.keys(),
			'idx_real' : dChan.values(),
			'featureImp': df['feature_importance_vals']
			}
		
		dfSummary = pd.DataFrame(data = dSummary)
		#goodidxoptuna = list(dfSummary['idx_real'])
		goodidxoptuna = list(dc)
		
		
		# dc = np.asarray(list(dictChann_lpltd.values()))
		# dc = dc[idx_2l]
		# dc2 = list(dc)
		
		# Good list pour optuna
		#dc1.extend(dc2)
		#goodidxoptuna = dc1
		#joblib.dump(goodidxoptuna,"idx_RFE_Opt.pkl")
	
		# Récupérer bons indices réels
		idxtoplot = association_df[association_df.idx.isin(goodidxoptuna)]['channels_reals']
		
		# idxtoplot2 = association_df[association_df.channels_th.isin(idxtoplot)]
		idxtoplot = association_df[association_df.channels_th.isin(idxtoplot)]
		idx_color = list(idxtoplot['channels_th'])
		#idx_size = list(df_toPlot_Imp[])		
		
		df_toPlot_Imp = idxtoplot.copy()
		l = list(df['feature_importance_vals'] * 1000)
		df_toPlot_Imp['feature_importance_vals']  = l
		
		
		# plot les bonnes couleurs selon RFE
		data_color = plotcol_RFE_DF(df_toPlot_Imp)
	
		# POUR PLOT LES LOBES
		plot_2D(data_color, data_chanTh, 1, 2, 'XXXXX', 2)
		
		
		# Indices réels associés aux indices théoriques
	#LFidx, LPidx, LTGidx, LTDidx, LOidx = extract_idx(LF, LP, LTG, LTD, LO, data_chanReal)
	#data_color = plotcol(LF, LP, LTG, LTD, LO,)
	#plot_2D(data_color, data_chanTh, 1, 2, 'LF, LP, LTD', 2)


	
