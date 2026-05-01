import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier
from utils.pre_processing import verificar_base, conf_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

'''
Descrição do Projeto: construção de um modelo de Machine Learning utilizando Árvore de Decisão para prever se um cliente terá uma pontuação de crédito alta,
indentificando impulsionadores do limite de crédito. Os dados foram limpos, 
transformados, balanceados e padronizados em https://github.com/Ema028/applied-analytics-cases/tree/master/analise_credit_score
O objetivo é comparar seu desempenho com o modelo de
Naive Bayes desenvolvido em https://github.com/Ema028/machine-learning/blob/master/naive_bayes usando a mesma base de dados
'''

X_treino = pd.read_csv('../data/base_credito_tratada/X_train.csv')
y_treino = pd.read_csv('../data/base_credito_tratada/y_train.csv')
X_teste = pd.read_csv('../data/base_credito_tratada/X_test.csv')
y_teste = pd.read_csv('../data/base_credito_tratada/y_test.csv')
verificar_base(X_treino, X_teste, y_treino, y_teste,  'Credit Score_High')
#nada que denuncie a variável alvo vazou para X_teste nem X_treino

''' 
para aplicar o algoritmo de árvore de decisão,treinamos o modelo com os dados de treino 
(critério de gini foi utilizado, por ser uma classificação binária)
ele aprende criando regras e dividindo os dados em grupos menores pelas características mais relevantes
depois, os dados de teste são usados para avaliar o desempenho, é possível melhorar o modelo ajustando hiperparâmetros 
e usando técnicas como pruning
'''

arvore_decisao = DecisionTreeClassifier(criterion='gini', random_state=0)
arvore_decisao.fit(X_treino, y_treino)

pred_treino = arvore_decisao.predict(X_treino)
acuracia_treino = accuracy_score(y_treino, pred_treino)
print(classification_report(y_treino, pred_treino))
print(f"\nAcurácia do modelo nos dados de treino: {acuracia_treino * 100:.2f}%\n")

pred_teste = arvore_decisao.predict(X_teste)
acuracia_teste = accuracy_score(y_teste, pred_teste)
print(classification_report(y_teste, pred_teste))
print(f"\nAcurácia do modelo nos dados de teste: {acuracia_teste * 100:.2f}%\n")

#como a acurácia deu 100%, decidi fazer a validação cruzada com 5 folds
#acurácia continua alta, média ~99%, indica que o modelo generaliza
X_total = pd.concat([X_treino, X_teste])
y_total = pd.concat([y_treino, y_teste])
scores = cross_val_score(arvore_decisao, X_total, y_total.values.ravel(), cv=5)
print(f"Acurácias obtidas em cada uma das 5 rodadas: {np.round(scores * 100, 2)}%")
print(f"Acurácia média: {scores.mean() * 100:.2f}%")

#como a acurácia foi igual tanto na base de treino quanto de teste, não parece ser um caso de overfitting

class_names = ['Crédito não alto', 'Crédito alto']
plt.figure(figsize=(12, 8))
plot_tree(arvore_decisao,
          filled=True,              #colore os nós de acordo com a classe
          rounded=True,
          feature_names=X_treino.columns,
          class_names=class_names)

plt.title("Visualização da Árvore de Decisão")
plt.show()

conf_matrix(y_teste, pred_teste, class_names)

profundidade = arvore_decisao.get_depth()
print(f"A profundidade da árvore é: {profundidade}")
'''
avaliação visual: 'Home Ownership_Rented' é a feature mais importante (nó raiz, maior redução de Gini)e depois vem 'Age'
abaixo tem um modelo reduzido apenas com elas para diminuir a complexidade da árvore e melhorar a generalização
'''

features_principais = ['Home Ownership_Rented', 'Age']
X_treino_reduzido = X_treino[features_principais]
#profundidade máxima ajustada pq o modelo estava se superajustando aos dados de treino e diminuir a complexidade
arvore_reduzida = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=2)
arvore_reduzida.fit(X_treino_reduzido, y_treino)

pred_red_treino = arvore_reduzida.predict(X_treino_reduzido)
acuracia_red_treino = accuracy_score(y_treino, pred_red_treino)
print(f"\nAcurácia do modelo reduzido nos dados de treino: {acuracia_red_treino * 100:.2f}%")

pred_red_teste = arvore_reduzida.predict(X_teste[features_principais])
acuracia_red_teste = accuracy_score(y_teste, pred_red_teste)
print(f"Acurácia do modelo reduzido nos dados de teste: {acuracia_red_teste * 100:.2f}%\n")
print(classification_report(y_teste, pred_red_teste))

plt.figure(figsize=(12, 8))
plot_tree(arvore_reduzida,
          filled=True,
          rounded=True,
          feature_names=features_principais,
          class_names=class_names)
plt.title("Árvore de Decisão apenas das 2 features principais")
plt.show()

conf_matrix(y_teste, pred_red_teste, class_names)

#a acurácia diminuiu um pouco, sendo ~94% na base de teste, enquanto no modelo original foi 100%, mas a árvore é visivelmente menor
#o desempenho da árvore reduzida foi melhor considerando a simplicidade, errando só 2 amostras em 33
#mas a reduzida teve queda de desempenho no recall, tendo 9% de falsos positivos no teste

'''
na validação, naive bayes obteve ~97% de acurácia, a árvore atingiu aproximadamente 99% na validação cruzada, indica boa capacidade de generalização
a árvore de decisão consegue identificar relações não lineares entre variáveis, enquanto naive bayes assume independência entre elas
o modelo reduzido mostrou que é possível manter um bom desempenho mesmo com poucas variáveis e é mais simples de explicar, 
por isso árvore de decisão apresentou melhor desempenho
'''