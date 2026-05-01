import xgboost as xgb
from utils.pre_processing import *


df = pd.read_csv("../data/carro.csv", delimiter=',')
print(df.info()) #tipos certos, única categórica é gender
data = Dataframe(df)

data.print_missing() #nada faltando
data.print_unique_values() #padronizado
print(data.df.describe().T) #sem sinal outlier
#data.box_plot(None, 'AnnualSalary', "Boxplot de Renda Anual") #dentro dos quadrantes já

data.drop_columns(['User ID']) #tirar id
encoders = data.label_encoding()
#print(data.df.info()) #ok, só numéricas

data.heatmap()
'''idade é o principal preditor(correlação 0.62), depois salário anual(0.36) 
genêro(-0.05) praticamente não vai influenciar o modelo'''
data.separar_base('Purchased')
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'Purchased')