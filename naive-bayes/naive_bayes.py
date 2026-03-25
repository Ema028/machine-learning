import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import cross_val_score
from utils.pre_processing import verificar_base

'''
Descrição do Projeto: construção de um modelo de Machine Learning para prever se um cliente terá uma pontuação de crédito alta,
indentificando impulsionadores do limite de crédito. Os dados foram limpos, 
transformados, balanceados e padronizados em https://github.com/Ema028/applied-analytics-cases/tree/master/analise_credit_score.
O objetivo principal ao aplicar o algoritmo de Naive Bayes é testar como 
um modelo baseado em probabilidades condicionais se sai ao tentar classificar o histórico numérico e categórico da base.
'''

X_treino = pd.read_csv('../data/base_credito_tratada/X_train.csv')
y_treino = pd.read_csv('../data/base_credito_tratada/y_train.csv')
X_teste = pd.read_csv('../data/base_credito_tratada/X_test.csv')
y_teste = pd.read_csv('../data/base_credito_tratada/y_test.csv')
verificar_base(X_treino, X_teste, y_treino, y_teste,  'Credit Score_High')

naive_bayes = GaussianNB()
#.values.ravel() usado para evitar warnings do pandas, garantir que o y seja um array unidimensional
naive_bayes.fit(X_treino, y_treino.values.ravel())

pred_train = naive_bayes.predict(X_treino)
acuracia = accuracy_score(y_treino, pred_train)
recall = recall_score(y_treino, pred_train, average='macro')

#em porcentagem
print(f"\nAcurácia: {acuracia * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

conf_matrix = confusion_matrix(y_treino, pred_train)
class_names = ['Crédito não alto', 'Crédito alto']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão (base de treino)', pad=15)
plt.ylabel('Valor Real', fontweight='bold')
plt.xlabel('Previsões', fontweight='bold')
plt.show()

pred_test = naive_bayes.predict(X_teste)
acuracia = accuracy_score(y_teste, pred_test)
recall = recall_score(y_teste, pred_test, average='macro')

#em porcentagem
print(f"\nAcurácia: {acuracia * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%\n\n")

conf_matrix = confusion_matrix(y_teste, pred_test)
class_names = ['Crédito não alto', 'Crédito alto']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão (base de teste)', pad=15)
plt.ylabel('Valor Real', fontweight='bold')
plt.xlabel('Previsões', fontweight='bold')
plt.show()
'''comparação treino e teste:
98.33% de acurácia no treino e 96.97% no teste, como a performance no teste foi superior ao treino,
os números altos não foram por causa de overfitting, no início cogitei que o modelo poderia olhar Credit Score_Low
e ter o gabarito, mas mesmo tirando essa coluna de X_teste e X_treino o desmpenho teve os mesmos números,
acredito que a pontuação perfeita seja pela previsibilidade e tamanho reduzido do dataset'''

'''resultados muito próximos de 100%, precisamos garantir que isso não 
foi sorte de uma divisão específica gerada pelo train_test_split.'''

#.predict_proba() devolve a porcentagem de certeza do modelo para cada classe
#retorna um array com duas colunas [probabilidade_classe_0(crédito não alto), probabilidade_classe_1(crédito alto)]
probabilidades = naive_bayes.predict_proba(X_teste)
prob_credito_alto = probabilidades[:, 1]
df_resultados = pd.DataFrame({
    'Valor Real': y_teste.values.ravel(),
    'Previsão': pred_test,
    'Certeza Credito Alto (%)': np.round(prob_credito_alto * 100, decimals= 2)})
print(df_resultados.head(10))

# Juntando treino e teste temporariamente para rodar validação cruzada na base inteira
X_total = pd.concat([X_treino, X_teste])
y_total = pd.concat([y_treino, y_teste])

scores_acuracia = cross_val_score(naive_bayes, X_total, y_total.values.ravel(), cv=5, scoring='accuracy')

print(f"\n\nAcurácias obtidas em cada uma das 5 rodadas: {np.round(scores_acuracia * 100, 2)}%")
print(f"Acurácia Média Final: {scores_acuracia.mean() * 100:.2f}%")
print(f"Desvio Padrão: {scores_acuracia.std() * 100:.2f}%")
'''acurácia se manteve alta em todas os folds, prova estatistica da hipótese inicial 
que o dataset é altamente previsível e não é um caso de overfitting ou leakage'''