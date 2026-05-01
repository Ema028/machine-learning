import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
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

modelo_xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth=4,
                               n_estimators=100, random_state=42, eval_metric='logloss')
modelo_xgb.fit(data.X_train, data.y_train)

previsoes = modelo_xgb.predict(data.X_test)
prob = modelo_xgb.predict_proba(data.X_test)
df_resultados = pd.DataFrame({'Previsão': previsoes,
                              'Probabilidade Compra': np.round(prob[:, 1] * 100, decimals= 2)})
print(df_resultados.head())

plt.figure(figsize=(12, 8))
sns.histplot(df_resultados['Probabilidade Compra'], bins=30)
plt.title('Distribuição de probabilidades de compra')
plt.show()

acuracia = accuracy_score(data.y_test, previsoes)
print(f"\nAcurácia do Modelo: {acuracia * 100:.2f}%\n")
print(classification_report(data.y_test, previsoes))
conf_matrix(data.y_test, previsoes, ['Não compra', 'compra'])
'''acurácia de 91.5%, precisão de 94% na classe1, 
mas poucos alarmes falsos, recall de 86%, deixa passar alguns clientes reais, 
mas de 96% para a classe0, eficiente em identificar quem não vai comprar'''