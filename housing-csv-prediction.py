# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Reading houses dataset

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print "\n\nO conjunto de dados de imoveis tem {} pontos " \
	  "com {} variaveis em cada.\n\n".format(*data.shape)

# Extracting statistics

minimum_price = np.amin(prices)
maximum_price = np.amax(prices)
mean_price = np.average(prices)
median_price = np.median(prices)
std_price = np.std(prices)

print "Estatisticas para os dados dos imoveis:\n"
print "Preco minimo: ${:,.2f}".format(minimum_price)
print "Preco maximo: ${:,.2f}".format(maximum_price)
print "Preco medio: ${:,.2f}".format(mean_price)
print "Preco mediano: ${:,.2f}".format(median_price)
print "Desvio padrao dos precos: ${:,.2f}".format(std_price)

# Definindo função de avaliação utilizando métrica Rˆ2. Best score = 1.0
def performance_metric(y_true, y_predict):
	return r2_score(y_true, y_predict)

# Misturando e separando os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 123)
print "\nSeparacao entre treinamento e teste feita com exito.\n"

def fit_model(X, y):
	# Gera conjuntos de validação-cruzada para o treinamento de dados
	cv_sets = ShuffleSplit(X.shape[0]		# qt total elementos
						, n_iter = 10 	# qt vezes embaralhar e dividir
						, test_size = 0.2
						, random_state = 123)

	grid = GridSearchCV(DecisionTreeRegressor()
						, dict(max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
						, make_scorer(performance_metric)
						, cv = cv_sets)

	# Encontrando os melhores parâmetros do estimador
	grid = grid.fit(X, y)

	return grid.best_estimator_

# Cria um regressor (DecisionTree) com o parâmetro 'max_depth
# otimizado para os dados de treinamento

regressor = fit_model(X_train, y_train)

print "O parametro 'max_depth' otimizado " \
      "para o modelo e {}.\n".format(regressor.get_params()['max_depth'])

# Parameters:
# RM: número médio de quartos entre os imóveis na vizinhança.
# LSTAT: porcentagem de proprietários na vizinhança considerados de “baixa renda”.
# PTRATIO: razão de estudantes para professores nas escolas de ensino fundamental e médio na vizinhança.
client_data = [[5, 17, 15], # Imovel 1
			   [4, 32, 22], # Imovel 2
			   [8, 3, 12]] # Imovel 3

# Mostra as estimativas
for i, price in enumerate(regressor.predict(client_data)):
	print "Preco estimado para o imovel {}: ${:,.2f}".format(i+1, price)
