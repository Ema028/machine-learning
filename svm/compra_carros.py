from sklearn.metrics import accuracy_score, classification_report
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


