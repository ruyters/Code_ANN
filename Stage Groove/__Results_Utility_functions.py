# -*- coding: utf-8 -*-

import pickle
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from __Viz_DNN_keras_Clf import * 
from __Viz_DNN_keras_Reg import * 
from __RedDim_DNN_Shapley import *


def load_results_pkl(filename):
	file = open(str(filename),'rb')
	varname = pickle.load(file, encoding="latin1")
	file.close()
	return varname

def save_results(filename, varname):
	f = open(str(filename),"wb")
	pickle.dump(resultat_groove,f)
	f.close()
	
	
'''
filename = "study_bestPart.pkl"
varname = 'study'

g1 = load_results_pkl('study_bestPart.pkl', varname)
g2 = load_results_pkl('res_groove_f70-99_bestPart.pkl', varname)

study = joblib.load("study_bestPart.pkl")

red = Color("red")
colors = list(red.range_to(Color("blue"), 73))
plot_DNN_clf(g1, 'groove', colors, p = 9, f = 70)

red = Color("red")
colors = list(red.range_to(Color("blue"), 27))
plot_DNN_clf(g2, 'groove', colors, p = 9, f = 27)


res_groove  = np.asarray(np.load('rslt_groove.npy', allow_pickle = True, fix_imports=True, encoding='ASCII'))
res_groove[0]
'''

	
def analysis_results(fd, ff, val, m):
	
	df = pd.DataFrame()
	for p in range (0, 30):
		for f in range(fd,ff):
			for n in range(0,3):
				for t in range(0, 10): 
					title = 'Shapley_FeatureImportance' + str(n) + "_p" + str(p) + "_t" + str(t) + "_f" + str(f) + ".pkl"
					# varname = "df_FI_p" + str(p) + "_f" + str(f) + "_t" + str(t)
	
					try: 
						df_add = joblib.load(title)
						df_add = df_add.assign(Part=p)
						df = merge_shapley_df(df, df_add)
					except FileNotFoundError:
						# print('File not found')		
						pass
	
	# df = pd.to_numeric(df['col_name'])	
	if m == True:
		grouped_df = df.groupby("col_name")
		mean_df = grouped_df.mean()
		mean_df = mean_df.sort_values(by = "feature_importance_vals", ascending = False)
		final_df = mean_df[0:val]
	else:
		return df
	# print(np.unique(df2))
	# print(df.value_counts())
	return final_df



def analysis_results_byPart(fd, ff, val, p):
	
	df = pd.DataFrame()
	for f in range(fd, ff):
		for n in range(0,3):
			for t in range(0, 10): 
				title = 'Shapley_FeatureImportance' + str(n) + "_p" + str(p) + "_t" + str(t) + "_f" + str(f) + ".pkl"
				# title = 'Shapley_FI_40ch' + str(n) + "_p" + str(p) + "_t" + str(t) + "_f" + str(f) + ".pkl"
				varname = "df_FI_p" + str(p) + "_f" + str(f) + "_t" + str(t)

				try: 
					df_add = joblib.load(title)
					df = merge_shapley_df(df, df_add)
				except FileNotFoundError:
					# print('File not found')		
					pass
	
	# df = pd.to_numeric(df['col_name'])	

	grouped_df = df.groupby("col_name")
	mean_df = grouped_df.mean()
	mean_df = mean_df.sort_values(by = "feature_importance_vals", ascending = False)
	final_df = mean_df[0:val]

	# print(np.unique(df2))
	# print(df.value_counts())
	return final_df


if __name__ == '__main__':
	import joblib
	from sklearn.metrics import ConfusionMatrixDisplay
	cm = joblib.load("CM_p2_t18_f6.pkl") # (n_outputs, 2, 2)
	
	cmd_obj = ConfusionMatrixDisplay(cm)
	cmd_obj.plot()
	plt.show()
	
	h = joblib.load("history2_t18_f6.pkl")
	
'''

#%% 
df_0015 = analysis_results(0, 15, 248, False)
df_1540 = analysis_results(15, 40, 248, False)
df_4080 = analysis_results(40, 80, 20, False)
df_8099 = analysis_results(80, 99, 20, False)

idx_0015 = np.int_(df_0015.index.values.tolist())
idx_1540 = np.int_(df_1540.index.values.tolist())
idx_4080 = np.int_(df_4080.index.values.tolist())
idx_8099 = np.int_(df_8099.index.values.tolist())

plt.bar(df_0015.index, df_0015['feature_importance_vals'])
#%% 
n = 100
fd, ff = 15, 40
df_0015_p2 = analysis_results_byPart(fd, ff, n, 2)
df_0015_p2['feature_importance_vals'] = MinMaxScaler().fit_transform(np.array(df_0015_p2['feature_importance_vals']).reshape(-1,1))
df_0015_p11 = analysis_results_byPart(fd, ff, n, 11)
df_0015_p11['feature_importance_vals'] = MinMaxScaler().fit_transform(np.array(df_0015_p11['feature_importance_vals']).reshape(-1,1))
df_0015_p16 = analysis_results_byPart(fd, ff, n, 16)
df_0015_p16['feature_importance_vals'] = MinMaxScaler().fit_transform(np.array(df_0015_p16['feature_importance_vals']).reshape(-1,1))
df_0015_p19 = analysis_results_byPart(fd, ff, n, 19)
df_0015_p16['feature_importance_vals'] = MinMaxScaler().fit_transform(np.array(df_0015_p16['feature_importance_vals']).reshape(-1,1))
df_0015_p22 = analysis_results_byPart(fd, ff, n, 22)
df_0015_p22['feature_importance_vals'] = MinMaxScaler().fit_transform(np.array(df_0015_p22['feature_importance_vals']).reshape(-1,1))

# df_0015_p28 = analysis_results_byPart(0, 15, 20, 28)

fig = plt.subplots(figsize =(20, 14))
plt.bar(df_0015_p16.index, df_0015_p16['feature_importance_vals'], color ="red")
plt.bar(df_0015_p2.index, df_0015_p2['feature_importance_vals'], color="red")
plt.bar(df_0015_p11.index, df_0015_p11['feature_importance_vals'], color ="blue")
plt.bar(df_0015_p19.index, df_0015_p19['feature_importance_vals'], color ="green")
plt.bar(df_0015_p22.index, df_0015_p22['feature_importance_vals'], color ="yellow")
plt.bar(df_0015_p28.index, df_0015_p28['feature_importance_vals'], color ="black")
plt.show()

fig = plt.subplots(figsize =(20, 14))
plt.scatter(df_0015_p16.index, df_0015_p16['feature_importance_vals'], color ="red")
plt.scatter(df_0015_p2.index, df_0015_p2['feature_importance_vals'], color="red")
plt.scatter(df_0015_p11.index, df_0015_p11['feature_importance_vals'], color ="blue")
plt.scatter(df_0015_p19.index, df_0015_p19['feature_importance_vals'], color ="green")
plt.scatter(df_0015_p22.index, df_0015_p22['feature_importance_vals'], color ="yellow")
plt.scatter(df_0015_p28.index, df_0015_p28['feature_importance_vals'], color ="black")
plt.show()


plt.scatter(df_0015_p2.index, df_0015_p2['feature_importance_vals'])
plt.scatter(df_0015_p11.index, df_0015_p11['feature_importance_vals'])


plt.bar(df_0015.index, df_0015['feature_importance_vals'])
plt.bar(df_1540.index, df_1540['feature_importance_vals'])
plt.bar(df_4080.index, df_4080['feature_importance_vals'])
plt.bar(df_8099.index, df_8099['feature_importance_vals'])
print(df_0015.index.values.tolist())
'''









