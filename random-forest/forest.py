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