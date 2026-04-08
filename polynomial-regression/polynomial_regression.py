from utils.pre_processing import *
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso

#mesma base utilizada para o modelo de regressão linear
df = pd.read_csv("../data/aluguel.csv", delimiter=';')
#apenas variáveis numéricas, sem valores faltando, sinais de outliers em Valor_Aluguel, Valor_Condominio e Metragem
#cauda muito longa em Valor_Aluguel e menos acentuadas nas outras duas
data = Dataframe(df)
data.drop_columns(['N_Suites'])

X = df[['Metragem']]
y = df['Valor_Aluguel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''capping não foi usado para não criar teto artificial, 
log não foi aplicado pra não deixar a curva dos dados muito reta e atrapalhar a regressão polinomial,
por isso RobustScaler por ser menos sensível a outliers'''
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
regressao_poly = LinearRegression()
regressao_poly.fit(poly_features.fit_transform(X_train), y_train)

y_pred = regressao_poly.predict(poly_features.transform(X_test))
print("Polinômio de grau 2(só com Metragem):")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\n")
'''R² de 0.5711, quase idêntico ao da regressão linear simples com log (R² de 0.5674). 
relação Metragem e Valor do Aluguel com teto de explicação em ~57%'''

poly_features4 = PolynomialFeatures(degree=4, include_bias=False)
regressao_poly4 = LinearRegression()
regressao_poly4.fit(poly_features4.fit_transform(X_train), y_train)

y_pred4 = regressao_poly4.predict(poly_features4.transform(X_test))
print("Polinômio de grau 4(só com Metragem):")
print(f"R²: {r2_score(y_test, y_pred4):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred4)):.4f}\n")
'''modelo de grau 4 teve um desempenho pior que o de grau 2, R² caiu (0.5711 para 0.5585) e 
o erro médio aumentou (~+30 reais), overfitting, modelo perdeu capacidade de generalização'''

#ordenar x e as previsões para consertar a curva no gráfico
X_plot = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_plot_poly2 = regressao_poly.predict(poly_features.transform(X_plot))
y_plot_poly4 = regressao_poly4.predict(poly_features4.transform(X_plot))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Dados Reais')
#linhas de regressão
plt.plot(X_plot, y_plot_poly2, color='blue', label='Regressão Grau 2', linewidth=2)
plt.plot(X_plot, y_plot_poly4, color='red', label='Regressão Grau 4', linewidth=2)
plt.title('Metragem vs Valor Aluguel')
plt.xlabel('Metragem(escalonada)')
plt.ylabel('Aluguel')
plt.legend()
plt.show()

data.separar_base('Valor_Aluguel')
data.robust_scaler()

multi_poly_features = PolynomialFeatures(degree=2, include_bias=False)
multi_poly_regression = LinearRegression()
X_train_poly = multi_poly_features.fit_transform(data.X_train)
X_test_poly = multi_poly_features.transform(data.X_test)
multi_poly_regression.fit(X_train_poly, data.y_train)
y_pred_multi = multi_poly_regression.predict(X_test_poly)

print("Polinômio de grau 2(com todas as features):")
print(f"R² do Polinômio: {r2_score(data.y_test, y_pred_multi):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(data.y_test, y_pred_multi)):.4f}")
'''depois de adicionar as outras características do imóvel, R² subiu (0.5711 para 0.6281), 
erro caiu (~-150 reais), melhorou a precisão da previsão'''

'''Obs: regressão Ridge encolhe os coeficientes de variáveis menos importantes, sem eliminar nenhuma variáveis
regressão Lasso zera os coeficientes de variáveis irrelevantes, 
Elastic Net combina a penalidade das duas regressões
foi feito um teste a seguir com Lasso como tentativa de lidar com o ruído'''

#escala dos valores de aluguel exigiu alpha alto para começar a zerar coeficientes
#ganho foi marginal, limite do que o modelo consegue extrair dos dados
regressao_lasso = Lasso(alpha=100.0, max_iter=10000, random_state=42)
regressao_lasso.fit(X_train_poly, data.y_train)
y_pred_lasso = regressao_lasso.predict(X_test_poly)

print("Modelo com Lasso:")
print(f"R²: {r2_score(data.y_test, y_pred_lasso):.4f}")
print(f"RMSE: R$ {np.sqrt(mean_squared_error(data.y_test, y_pred_lasso)):.2f}\n")

nomes_features = multi_poly_features.get_feature_names_out(data.X_train.columns)
coeficientes = regressao_lasso.coef_

features_mantidas = []
features_zeradas = []

for nome, coef in zip(nomes_features, coeficientes):
    if coef == 0:
        features_zeradas.append(nome)
    else:
        features_mantidas.append((nome, coef))

print(f"Total de features criadas pelo polinômio: {len(nomes_features)}")
print(f"Features que o Lasso considerou inúteis e zerou: {len(features_zeradas)}")
print(f"Features que o Lasso manteve: {len(features_mantidas)}\n")