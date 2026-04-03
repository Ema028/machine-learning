from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from utils.pre_processing import *

df = pd.read_csv("../data/wine.csv", delimiter=',')
print(df.info()) #só váriaveis numéricas
data = Dataframe(df)
data.print_missing() #nada faltando
print(df.describe().T) #sinais de outliers em residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide e sulphates
#75% muito longe de máximo e assimetria na média

data.histogram('residual sugar', "Distribuição de residual sugar")
data.histogram('chlorides', "Distribuição de chlorides")
data.histogram('free sulfur dioxide', "Distribuição de free sulfur dioxide")
data.histogram('total sulfur dioxide', "Distribuição de total sulfur dioxide")
data.histogram('sulphates', "Distribuição de sulphates")
#todos com cauda muito longa à direita

data.capping_outliers(['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates'])
print(df['quality'].value_counts()) #verificação balanceamento da váriavel target
data.heatmap()

#variáveis com maior correlação
colunas_corr_forte = ['alcohol', 'volatile acidity', 'sulphates', 'quality']
df_reduzido = df[colunas_corr_forte]
print(df_reduzido.head())

X = df_reduzido.drop('quality', axis=1)
y = df_reduzido['quality']
#separando df_reduzido, pra comparar resultados do modelo com o df original
X_train_reduzido, X_test_reduzido, y_train_reduzido, y_test_reduzido = train_test_split(X, y, test_size=0.2, random_state=42)

data.separar_base('quality')

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(data.X_train, data.y_train)
y_pred = random_forest.predict(data.X_test)

random_forest_reduzido = RandomForestClassifier(random_state=42)
random_forest_reduzido.fit(X_train_reduzido, y_train_reduzido)
y_pred_reduzido = random_forest_reduzido.predict(X_test_reduzido)

print(f"Acurácia do modelo nos dados de teste: {accuracy_score(data.y_test, y_pred) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred)}")

print(f"Acurácia do modelo reduzido nos dados de teste: {accuracy_score(y_test_reduzido, y_pred_reduzido) * 100:.2f}%\n")
print(f"Relatório de Classificação do modelo reduzido:\n{classification_report(y_test_reduzido, y_pred_reduzido)}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(data.y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=random_forest.classes_,
            yticklabels=random_forest.classes_)
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão')
plt.show()
''''o modelo teve acurácia de ~64%, com as classes mais comuns (5 e 6) tendo bons F1-Scores (~0.71 e ~0.64)
enquanto o modelo reduzido teve um desempenho muito parecido (~62%)

o modelo falhou em prever as classes mais raras (3, 4 e 8), zerou o F1-Score nelas, 
relação direta com o desbalanceamento dos dados (notas 5 e 6 eram ~82% da base)
algoritmo priorizou acertos nas classes mais frequentes'''

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
'''otimização com Random Search elevou a acurácia de ~64.38% para ~66.25%, F1-Score também teve um leve ganho
modelo continua com F1-Score igual a 0 para as classes minoritárias
redefinir o espaço do problema e fazer o balanceamento pode melhorar as previsões'''

#binning para agrupar variáveis contínuas em categorias comportamentais
bins = [2, 4, 6, 8]
labels = ['Ruim', 'Médio', 'Bom']
data.y_train = pd.cut(data.y_train, bins=bins, labels=labels)
data.y_test = pd.cut(data.y_test, bins=bins, labels=labels)

#balanceando as classes do alvo com amostras sintéticas
data.smote()
random_forest_upgrade = RandomForestClassifier(**random_search.best_params_, random_state=42)
random_forest_upgrade.fit(data.X_train, data.y_train)
y_pred_upgrade = random_forest_upgrade.predict(data.X_test)

print(f"Acurácia (com Binning + SMOTE): {accuracy_score(data.y_test, y_pred_upgrade) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred_upgrade)}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(data.y_test, y_pred_upgrade), annot=True, fmt='d', cmap='Greens',
            xticklabels=random_forest_upgrade.classes_,
            yticklabels=random_forest_upgrade.classes_)
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão (Classes Agrupadas)')
plt.show()
'''Análise do Modelo Pós-Upgrade 
acurácia global saltou de ~66% para ~85.9%
classe 'Ruim', era ignorada pelo modelo (Recall e F1 zerados), agora identifica 45% (Recall 0.45)
classe 'Bom' atingiu um Recall de 85%, modelo identifica e mantém a maioria dos casos de alta qualidade, F1-Score (0.73)
classe 'Médio' com performance muito alta(F1-Score 0.91)
'''