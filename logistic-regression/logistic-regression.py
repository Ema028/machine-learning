from utils.pre_processing import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#o objetivo é construir um modelo de regressão capaz de indicar se novos pacientes estão propensos a doenças cariovasculares
df = pd.read_csv("../data/cardio.csv", delimiter=';')

print(df.info())
#(gender: 1-mulher, 2-homem)(true= 1, false= 0)
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

data = Dataframe(df)
data.print_missing() #valores faltando em weight 0.24% fração insignificante
df.dropna(subset=['weight'], inplace=True)

print(df.describe().T) #height com máximos e mínimos estranhos, recem-nascido de 70cm? e pessoa gigante de 250cm
data.histogram('height', "distribuição de altura") #eixo x muito esticado, min e max são pontos isolados
data.capping_outliers('height')
data.histogram('weight', "distribuição de peso") #cauda muito longa para direita
data.capping_outliers('weight')

data.heatmap()
#alvo é cardio_disease, variáveis com maior correlação são age e cholesterol, seguidas por weight
#gluc tem a maior correlação com cholesterol, faz sentido, pacientes com alto colesterol têm tendência a também ter glicose alta
#fumantes têm hábito de beber também e vice-versa

#qual relação entre idade e doenças cardíacas?
data.box_plot('cardio_disease', 'age', "Doenças Cardíacas por idade")
#pacientes mais velhos têm chances maiores de problemas cardiovasculares

#qual relação entre doenças cardíacas e colesterol?
data.box_plot('cardio_disease', 'cholesterol', "Doenças Cardíacas por Colesterol")
#esmagadora maioria dos pacientes sem doença tem colesterol no nível 1, caixa achatada embaixo
#maior dispersão entre os com problemas

#qual relação entre doenças cardíacas e peso?
data.box_plot('cardio_disease', 'weight', "Doenças Cardíacas por peso")
#pacientes com doença tendem a ser mais pesados

data.separar_base('cardio_disease')
data.std_scaler() #regressão logística é sensível à escala das variáveis, padronizar garante que variáveis com valores altos não dominem o modelo
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'cardio_disease') #dados já balanceados

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(data.X_train, data.y_train)

intercept = log_reg.intercept_[0]
print(f"\nIntercept: {intercept:.4f}\n")

#peso que o modelo deu para cada variável, com os dados padronizados dá comparar esses valores diretamente
coeficientes = log_reg.coef_[0]
df_coef = pd.DataFrame({
    'Variável': data.X_train.columns,
    'Coeficiente': coeficientes
})
df_coef = df_coef.sort_values(by='Coeficiente', ascending=False)
print(df_coef.to_string(index=False))
print("\n")

previsoes = log_reg.predict(data.X_test)
acuracia = accuracy_score(data.y_test, previsoes)
print(f"Acurácia do Modelo: {acuracia * 100:.2f}%\n")
print(classification_report(data.y_test, previsoes))

'''
regressão logística prevê a probabilidade de um evento ocorrer, pega os dados e mapeia entre 0 e 1 e 
usa essa probabilidade para categorizar os dados em classes(modelo de classificação)
assim como a linear calcula coeficientes para cada variável e um intercepto formando uma equação linear simples, 
a diferença é que a logística aplica a função sigmoide no resultado da equação
'''