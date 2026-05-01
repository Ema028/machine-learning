from utils.pre_processing import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("../data/campaign.csv", delimiter=';')
print(df.info()) #tipos certos, 2 categóricas
data = Dataframe(df)

data.print_missing() #income com 1.071429% de valores faltando(24), pouco expressivo
data.drop_missing()

data.print_unique_values()
status_mapping = {
    'Married': 'Partner',
    'Together': 'Partner',
    'Divorced': 'Single',
    'Widow': 'Single',
    'Alone': 'Single',
    'Absurd': 'Single',
    'YOLO': 'Single'
}
data.df['Marital_Status'] = df['Marital_Status'].replace(status_mapping)

education_mapping = {
    'Basic': 'Undergraduate',
    'Graduation': 'Graduate',
    'Master': 'Postgraduate',
    '2n Cycle': 'Postgraduate',
    'PhD': 'Postgraduate'
}
data.df['Education'] = df['Education'].replace(education_mapping)

print(data.df.describe().T) #colunas de gastos com cauda longa
data.box_plot_multi(['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],"Gastos Anuais")
data.apply_log(['MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])

data.box_plot(None, 'Income', "Boxplot de Renda Anual") #ponto muito longe do resto
data.box_plot(None, 'Year_Birth', "Boxplot de nascimento") #poucas datas muito antigas(seres imortais)
data.capping_outliers(['Year_Birth', 'Income'])

data.heatmap()
data.one_hot()
data.one_hot_heatmap()

#funil de conversão funciona?
data.bar_plot('NumWebVisitsMonth', 'WebPurchases', "Compras por Visitas no Site")
'''pico de conversão com duas visitas mensais, alta intenção de compra
quarta visita pra frente, a taxa caí progressivamente e de treze pra frente zera, repique em nove e dez visitas, provavelmente caçadores de promoções
maior chance de fechar a venda nas três primeiras interações'''

#concentração de clientes
data.bar_plot('NumStorePurchases', 'WebPurchases', 'Compras em Loja Física vs Web')
'''não tem canibalização, clientes com até três compras na loja física pouca conversão online,
a partir da quarta ida à loja taxa de conversão alta, cliente converte-se naturalmente em um comprador digital'''

data.separar_base('WebPurchases')
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'WebPurchases') #balanceado já
data.std_scaler()

log_reg = LogisticRegression(max_iter=1000) #modelo base
log_reg.fit(data.X_train_scalled, data.y_train)

intercept = log_reg.intercept_[0]
print(f"\nIntercept: {intercept:.4f}\n")

#peso que o modelo deu para cada variável, com os dados padronizados dá comparar esses valores diretamente
coeficientes = log_reg.coef_[0]
df_coef = pd.DataFrame({
    'Variável': data.X_train_scalled.columns,
    'Coeficiente': coeficientes
})
df_coef = df_coef.sort_values(by='Coeficiente', ascending=False)
print(df_coef.to_string(index=False))
'''alto consumo de vinhos e número de visitas ao site são os maiores impulsionadores de compras na web
os pessoas mais jovens, solteiras e com filhos, têm menor conversão online'''

previsoes = log_reg.predict(data.X_test_scalled)
acuracia = accuracy_score(data.y_test, previsoes)
print(f"\nAcurácia do Modelo: {acuracia * 100:.2f}%\n")
print(classification_report(data.y_test, previsoes))
'''86% de acurácia e recall alto de 93% para positivos, o algoritmo é bom em identificar potencial compradores'''

previsoes_proba = log_reg.predict_proba(data.X_test_scalled)[:, 1] #tds as linhas da coluna de indice 1(probabilidade de cada previsão)
plt.figure(figsize=(8, 6))

fpr, tpr, thresholds = roc_curve(data.y_test, previsoes_proba) #fpr(taxa falsos positivos) tpr(taxa de verdadeiros positivos) threshold(cutoff de qtns % de certeza eu tenho)
auc_score = roc_auc_score(data.y_test, previsoes_proba)

plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (área = {auc_score:.2f})')
#linha de referência do chute aleatório
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
#auc-roc de 91%

class_names = ['Não comprador web', 'Comprador web']
conf_matrix(data.y_test, previsoes, class_names)

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(data.X_train, data.y_train)
y_pred = random_forest.predict(data.X_test)

print(f"Acurácia do modelo nos dados de teste: {accuracy_score(data.y_test, y_pred) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred)}")

param_grid = {
    'n_estimators': [50, 100, 200],            #n_arvores
    'max_depth': [None, 10, 20, 30],           #profundidade máxima
    'min_samples_split': [2, 5, 10],           #mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],             #mínimo de amostras em nó folha
    'max_features': ['sqrt', 'log2', None]     #n_caracteristicas
}

random_forest_base = RandomForestClassifier(random_state=42)
#100 combinações aleatórias, 5 folds de validação cruzada
random_search = RandomizedSearchCV(
    estimator=random_forest_base,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    verbose=2,          #mostra o progresso no terminal
    random_state=42,
    n_jobs=-1           #todos os núcleos do processador para ser mais rápido
)

random_search.fit(data.X_train, data.y_train)
melhor_modelo = random_search.best_estimator_
print(f"\nMelhores Hiperparâmetros: {random_search.best_params_}\n")

y_pred_rs = melhor_modelo.predict(data.X_test)
print(f"Acurácia depois de Random Search: {accuracy_score(data.y_test, y_pred_rs) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred_rs)}")
'''tinha 90.99% de acurácia antes e foi refinado para 91.44% depois do random search, além de um recall de 95% para o positivo
sendo superior e cometendo menos erros que a regressão logística nesse caso'''

conf_matrix(data.y_test, y_pred_rs, class_names)