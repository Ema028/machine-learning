from utils.pre_processing import *
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

'''
o objetivo desse modelo é diagnosticar e classificar pacientes
para doenças da tireoide com base em perfis demográficos e resultados de exames 
clínicos (TSH, T3, TT4, FTI). 
'''
df = pd.read_csv("../data/hypothyroid.csv", delimiter=',')
data = Dataframe(df)
print(data.df.info()) #só categóricas

data.to_number(['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'])
data.print_missing()
data.drop_columns(['TBG', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured',
                   'FTI measured', 'TBG measured', 'referral source'])
#TBG falta tudo, não traz informação e colunas de measured só diz se foi feito exame
#'referral source' estava causando ruído, dado admnistrativo
data.df.dropna(subset=['age'], inplace=True) #só 1 linha sem idade, remover ela
#poucos faltantes ~5%, knn imputer estava causando data leakage(substituia faltantes aprendendo valores da base toda), decidi usar simple imputer
data.imputar_simples(['TSH', 'T3', 'TT4', 'T4U', 'FTI'])

data.print_unique_values()
#remapeando booleanas
colunas_booleanas = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                     'sick', 'pregnant', 'thyroid surgery', 'I131 treatment',
                     'query hypothyroid', 'query hyperthyroid', 'lithium',
                     'goitre', 'tumor', 'hypopituitary', 'psych']
data.df[colunas_booleanas] = data.df[colunas_booleanas].replace({'f': 0, 't': 1}).astype(int)
data.df['binaryClass'] = data.df['binaryClass'].replace({'N': 0, 'P': 1}).astype(int)
data.df['sex'] = data.df['sex'].replace({'?': np.nan, 'F': 0, 'M': 1}) #150 faltando(~4%)
data.imputar_simples(['sex'], estrategia='most_frequent')
data.df['sex'] = data.df['sex'].astype(int)

print(data.df.describe().T) #sinais de outlier em 'age', 'TSH', 'TT4', 'FTI'
data.box_plot_multi(['age', 'TSH', 'TT4', 'FTI'], "Antes de Tratar Outliers")
data.apply_log(['TSH', 'TT4', 'FTI']) #cauda longa bilateral, log para melhorar escala
data.capping_outliers(['age']) #só 1 valor mt distoante(vampiro de 455 anos)
data.box_plot_multi(['TSH', 'TT4', 'FTI'], "Depois de Tratar Outliers")
'''melhor análise visual, caudas longas ainda lá, mas capping evitado para não descaracterizar os exames
o q faz sentido pela escolha do algoritmo de árvores que não é tão sensível a outliers e 
não precisam de escalonamento(por isso std scaler não usada)'''

#print(data.df.info())
data.heatmap() #binaryClass têm forte correlação negativa com tsh(-0.69)
data.separar_base('binaryClass')
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'binaryClass')
#base muito desbalanceada, doenças da tireoide são raras (apenas 6.75% na base de teste), treinar o modelo a não ignorar a minoria com oversampling
data.smote()

#decision tree como modelo baseline
'''na primeira tentativa teve muito sobreajuste por isso profundidade limitada'''
arvore_decisao = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=4)
arvore_decisao.fit(data.X_train, data.y_train)

pred_treino = arvore_decisao.predict(data.X_train)
acuracia_treino = accuracy_score(data.y_train, pred_treino)
print(classification_report(data.y_train, pred_treino))
print(f"\nAcurácia do modelo base nos dados de treino: {acuracia_treino * 100:.2f}%\n")

pred_teste = arvore_decisao.predict(data.X_test)
acuracia_teste = accuracy_score(data.y_test, pred_teste)
print(classification_report(data.y_test, pred_teste))
print(f"\nAcurácia do modelo base nos dados de teste: {acuracia_teste * 100:.2f}%\n")

#validação cruzada
X_total = pd.concat([data.X_train, data.X_test])
y_total = pd.concat([data.y_train, data.y_test])
scores = cross_val_score(arvore_decisao, X_total, y_total.values.ravel(), cv=5)
print(f"Acurácias do modelo base obtidas em cada uma das 5 rodadas: {np.round(scores * 100, 2)}%")
print(f"Acurácia média: {scores.mean() * 100:.2f}%")
#modelo generaliza

class_names = ['Saudável', 'Doente']
plt.figure(figsize=(12, 8))
plot_tree(arvore_decisao, filled=True, rounded=True,
          feature_names=data.X_train.columns, class_names=class_names)
plt.title("Visualização da Árvore de Decisão")
plt.show()

conf_matrix(data.y_test, pred_teste, class_names)
profundidade = arvore_decisao.get_depth()
print(f"A profundidade da árvore é: {profundidade}")

data.feature_importance(arvore_decisao) #só 6 features não nulas, ~96% do resultado depende do exame de tsh
features_reduzidas = ['TSH', 'on thyroxine', 'TT4', 'FTI', 'thyroid surgery', 'query hypothyroid']
X_treino_red = data.X_train[features_reduzidas]
X_teste_red = data.X_test[features_reduzidas]

parametros_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4],
                   'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0),
                           param_grid=parametros_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_treino_red, data.y_train)
melhor_arvore_red = grid_search.best_estimator_

print(f"Melhores Hiperparâmetros:\n{grid_search.best_params_}")
pred_red = melhor_arvore_red.predict(X_teste_red)
acuracia_red_teste = accuracy_score(data.y_test, pred_red)
print(f"Acurácia do modelo base reduzido nos dados de teste: {acuracia_red_teste * 100:.2f}%\n")
print(classification_report(data.y_test, pred_red))

plt.figure(figsize=(12, 8))
plot_tree(melhor_arvore_red, filled=True, rounded=True,
          feature_names=features_reduzidas, class_names=class_names)
plt.title("Árvore de Decisão apenas com as features principais")
plt.show()
conf_matrix(data.y_test, pred_red, class_names)
#taxa de positivos verdadeiros e falsos verdadeiros igual mais com 15 features a menos, acurácia também
'''modelo provou matematicamente que usar apenas 6 features chave gera o mesmo resultado que 21 variáveis,
o que significa, redução de custos laboratoriais, menos formulários e um diagnóstico mais rápido'''

#modelo xgboost para classificação binária, vale o aumento de complexidade?
parametros_random = {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 4, 5, 6],
                     'n_estimators': [50, 100, 200, 300], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]}

random_search = RandomizedSearchCV(XGBClassifier(random_state=0),
                                       param_distributions=parametros_random, n_iter=20, cv=5,
                                       scoring='accuracy', n_jobs=-1,  random_state=42)

random_search.fit(X_treino_red, data.y_train)
modelo_xgb = random_search.best_estimator_

previsoes = modelo_xgb.predict(X_teste_red)
prob = modelo_xgb.predict_proba(X_teste_red)
auc_roc(data.y_test, prob)
df_resultados = pd.DataFrame({'Previsão': previsoes,
                              'Probabilidade Doença': np.round(prob[:, 1] * 100, decimals= 2)})
print(df_resultados.head())

plt.figure(figsize=(12, 8))
sns.histplot(df_resultados['Probabilidade Doença'], bins=30)
plt.title('Distribuição de probabilidades de doença')
plt.show()
'''modelo atinge ~99.9% de certeza nos diagnósticos positivos
se a chance cair na faixa de 40% a 60%, o sistema poderia acionar uma revisão médica manual'''


acuracia = accuracy_score(data.y_test, previsoes)
print(f"\nAcurácia do Modelo XGBoost: {acuracia * 100:.2f}%\n")
print(classification_report(data.y_test, previsoes))
conf_matrix(data.y_test, previsoes, class_names)

data.feature_importance(modelo_xgb, colunas=X_teste_red.columns) #TSH tem ~80% de influência sozinho
'''árvore venceu na acurácia por uma margem mínima (99.47% vs 99.34%), mas concentra 95% da decisão só no TSH 
XGBoost reduziu a dependência para ~80% e passou a considerar mais o histórico do paciente ('on thyroxine' com 12.5%)'''