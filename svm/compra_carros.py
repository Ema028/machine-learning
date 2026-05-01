from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from utils.pre_processing import *

df = pd.read_csv("../data/carro.csv", delimiter=',')
print(df.info())
data = Dataframe(df) #mesmos dados de xgboost, já sem outliers, padronizados e sem valor faltando
data.drop_columns(['User ID']) #tirar id
encoders = data.label_encoding()

#data.heatmap() -> mesmo mapa de correlação de xgboost/graficos/heatmap.png
'''idade é o principal preditor(correlação 0.62), depois salário anual(0.36) 
genêro(-0.05) praticamente não vai influenciar o modelo'''

data.separar_base('Purchased')
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'Purchased')
data.std_scaler()

svm_linear = SVC(kernel='linear', random_state=42, probability=True)
svm_linear.fit(data.X_train_scalled, data.y_train)
pred_linear = svm_linear.predict(data.X_test_scalled)

print(f"Acurácia(kernel linear): {accuracy_score(data.y_test, pred_linear) * 100:.2f}%\n")
print(f"Relatório de Classificação(kernel linear):\n{classification_report(data.y_test, pred_linear)}")
class_names = ['Não compra', 'compra']
conf_matrix(data.y_test, pred_linear, class_names)

svm_poly = SVC(kernel='poly', random_state=42, probability=True)
svm_poly.fit(data.X_train_scalled, data.y_train)
pred_poly = svm_poly.predict(data.X_test_scalled)

print(f"Acurácia(kernel poly): {accuracy_score(data.y_test, pred_poly) * 100:.2f}%\n")
print(f"Relatório de Classificação(kernel poly):\n{classification_report(data.y_test, pred_poly)}")
conf_matrix(data.y_test, pred_poly, class_names)
'''svm com kernel poly se saiu melhor, acurácia subiu de 80.50% para 82.00%
e na classe de compradores aumentou tanto a precisão (de 89% para 91%) quanto o recall (de 64% para 66%)
linha que separa os compradores dos não-compradores nos seus dados não é perfeitamente reta -> adicionar curvatura (polinômios) ajudou o algoritmo'''
'''XGBoost obteve de longe as melhores previsões comparado com os dois
91.5% de acurácia geral, recall de 86% e precisão de 94% na classe 1'''


