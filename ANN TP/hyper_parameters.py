
####################################################################
# Dictionnaires d'hyperparamètres 
####################################################################

################################################################################################
# COMBINAISON 1 : TAUX D'APPRENTISSAGE 

params_1 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.000001,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_2 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.00001,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_3 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_4 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_5 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.01,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_6 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.1,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_6 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 1,
	'hidden_1' : 100,
	'hidden_2' : 100,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

list_tt = [params_1, params_2, params_2, params_3, params_4, params_5, params_6]

################################################################################################
# COMBINAISON 2 : NOMBRE DE NEURONES PAR COUCHE AVEC ETA MOYEN

params_1 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 64,
	'hidden_2' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_2 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_3 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 64,
	'hidden_2' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_4 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}


list_tt = [params_1, params_2, params_2, params_3, params_4]

################################################################################################
# COMBINAISON 3 : BATCH SIZE
# TODO : DEFINIR HIDDEN_1 et HIDDEN_2

	params_1 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_2 = {
	'batch_size' : 10,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_3 = {
	'batch_size' : 15,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_4 = {
	'batch_size' : 20,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}


################################################################################################
# COMBINAISON 4 : TROIS COUCHES CACHEES

params_1 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 64,
	'hidden_3' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_2 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 128,
	'hidden_3' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_3 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 256,
	'hidden_2' : 256,
	'hidden_3' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_4 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 64,
	'hidden_2' : 64,
	'hidden_3' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_5 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 64,
	'hidden_2' : 128,
	'hidden_3' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

params_6 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.0001,
	'hidden_1' : 64,
	'hidden_2' : 256,
	'hidden_3' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}


list_tt = [params_1, params_2, params_2, params_3, params_4, params_5, params_6]


################################################################################################
# COMBINAISON 5 : TROIS COUCHES CACHEES V2


params_1 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 256,
	'hidden_2' : 256,
	'hidden_3' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_2 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 128,
	'hidden_2' : 128,
	'hidden_3' : 128,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_3 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 64,
	'hidden_2' : 64,
	'hidden_3' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_4 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 256,
	'hidden_2' : 64,
	'hidden_3' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_5 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 256,
	'hidden_2' : 128,
	'hidden_3' : 256,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}

	params_6 = {
	'batch_size' : 5,
	'nb_epochs' : 10,
	'eta' : 0.001,
	'hidden_1' : 64,
	'hidden_2' : 256,
	'hidden_3' : 64,
	'loss_func' : torch.nn.MSELoss(reduction='sum'),
}