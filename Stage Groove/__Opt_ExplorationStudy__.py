# -*- coding: utf-8 -*-

from __Opt_Optuna_Clf__ import *

def plot_study(study):
	'''
	Parameters
	----------
	study : study

	Returns
	-------
	None. '''
	optuna.visualization.matplotlib.plot_param_importances(study)
	optuna.visualization.matplotlib.plot_optimization_history(study)
	optuna.visualization.matplotlib.plot_slice(study)
	optuna.visualization.matplotlib.plot_slice(study, ['learning_rate', 'freq'])
	optuna.visualization.matplotlib.plot_contour(study, ['learning_rate', 'optimizer'])
	optuna.visualization.matplotlib.plot_contour(study, ['freq', 'part'])
	optuna.visualization.matplotlib.plot_contour(study, ['learning_rate','freq'])
	optuna.visualization.matplotlib.plot_contour(study, ['freq','batch'])
	optuna.visualization.matplotlib.plot_contour(study, ['freq','optimizer'])
	optuna.visualization.matplotlib.plot_intermediate_values(study)

def print_study(study):
	trial = study.best_trial
	pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
	
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))
	print("Best trial:")
	trial = study.best_trial
	print("  Value: ", trial.value)
	
	
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
	
	number_of_trials = len(study.trials)
	first_trial = study.trials[0]
	best_score_so_far = first_trial.value
	
	for report_index in range(1, number_of_trials):
	    trial_to_report = study.trials[report_index]
	    score_of_trial_to_report = trial_to_report.value
	    improved_score = (score_of_trial_to_report <= best_score_so_far)
	    if improved_score:
	        best_score_so_far = score_of_trial_to_report
	        print('\nTrial {}:'.format(trial_to_report.number), end=' ')
	        print('began at {}.'.format(trial_to_report.datetime_start))
	        print('Score was {},'.format(trial_to_report.value), end=' ')
	        print('and its parameters were: {}\n'.format(trial_to_report.params))
			
def select_df(df, score):
	df = df[df['value'] == score]
	return df 

def merge_df(df_names):
	dff = pd.DataFrame()
	
	for dfn in df_names:
		study = joblib.load(dfn)
		df = study.trials_dataframe()
		df = select_df(df, 0.0)
		dff = dff.append(df)
	
	return dff

# Trials storage
study_name = "OptRes_Binary_AllData_fHG.pkl"
study = joblib.load(study_name)
df = study.trials_dataframe()
df = select_df(df, 0)
plot_study(study)
'''
df_names = ['OptRes_Binary_AllPart_f60900_t50.pkl', 'OptRes_Binary_AllPart-7_f60900_t150.pkl',
			'OptRes_Binary_AllPart-7-11_f60900_t30.pkl', 'OptRes_Binary_AllPart-7-11-2_f60900_t30.pkl',
			'OptRes_Binary_AllPart-7-11-2-16_f60900_t30.pkl', 'OptRes_Binary_AllPart-7-11-2-16-28_f6090_t100.pkl']

dff = merge_df(df_names)
# print_study(study)

dff = pd.DataFrame()
dfff=dff.append(df)
'''
''' STUDY NAMES
OptRes_Binary_AllPart_f60900_t50.pkl
OptRes_Binary_AllPart-7_f60900_t150.pkl
OptRes_Binary_AllPart_7-11_f60900_t30.pkl
OptRes_Binary_AllPart_7-11-2_f60900_t30.pkl
OptRes_Binary_AllPart_7-11-2-16_f60900_t30.pkl
OptRes_Binary_AllPart_7-11-2-16-28_f60900_t100.pkl
'''
