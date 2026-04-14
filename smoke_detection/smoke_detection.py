from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from utils.pre_processing import *

df = pd.read_csv("../data/smoke_detection.csv", delimiter=',')
print(df.info()) #só variáveis numéricas
data = Dataframe(df)
data.print_missing() #nada faltando
data.drop_columns(['Unnamed: 0']) #desconsiderando indice da linha
#notação científica tava atrapalhando visualização de outliers
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print(data.df.describe().T)
#sinais de outliers nas colunas:
colunas_outliers = ['TVOC[ppb]', 'eCO2[ppm]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5']
for coluna in colunas_outliers:
    data.box_plot(coluna, None, f'Distribuição: {coluna}', x=8, y=6)
data.apply_log(colunas_outliers) #faz sentido para cauda muito longa, muitos outliers e regressão logística como baseline
"""
como é um problema de classificação binária,incêndio ou não incêndio, 
regressão logística usada como baseline para provar se um algoritmo mais complexo vale a pena
random forest escolhido pelas variáveis em escalas heterogêneas e 
multicolinearidade presente entre os sensores, bom para mapear picos não-lineares de fumaça e 
calcula a importância das variáveis -> permite identificar sensores redundantes
"""
data.separar_base('Fire Alarm', columns=['Fire Alarm', 'UTC', 'CNT']) #removidas coluna de contagem de tempo e amostras pq modelo estava usando de gabarito
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'Fire Alarm')
data.smote() #balancear dados de treino
data.std_scaler() #escalonamento por causa da regressão

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(data.X_train_scalled, data.y_train)

previsoes = log_reg.predict(data.X_test_scalled)
acuracia = accuracy_score(data.y_test, previsoes)
print(f"\nAcurácia da regressão logistica: {acuracia * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, previsoes)}")

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(data.X_train, data.y_train)
y_pred = random_forest.predict(data.X_test)

print(f"Acurácia de random forest: {accuracy_score(data.y_test, y_pred) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred)}")
'''
random search foi retirado do código por não ter espaço para melhora,
métricas de random forest já maximizadas, só causa risco de overfitting
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
print(f"Acurácia do melhor modelo nos dados de teste: {accuracy_score(data.y_test, y_pred_rs) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred_rs)}")'''