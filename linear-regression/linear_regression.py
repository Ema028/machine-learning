from utils.pre_processing import *
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/aluguel.csv", delimiter=';')
print(df.info())  #apenas variáveis numéricas
data = Dataframe(df)
data.print_missing() #sem valor faltando
print(df.describe().T)
'''Valor_Aluguel, Valor_Condominio e Metragem, talvez outliers, pois Q3 muito diferente do valor máximo
variáveis quantitativas (quartos, banheiros, suítes e vagas) com máximos altos, 
mas faz sentido para imóveis de alto padrão'''

data.histogram('Valor_Aluguel', "Distribuição de preço do aluguel")
#Valor_Aluguel com valores extremos muito acentuados, cauda muito longa, muita assimetria

data.histogram('Valor_Condominio', "Distribuição de preço do condomínio")
#cauda longa também, menos que aluguel, bastante disperção

data.histogram('Metragem', "Distribuição de metragem do imóvel")
#cauda longa também, distorção, mas menos que as outras

'''tentei tratamento com capping pelo IQR, mas fez um teto artificial e para não remover dados usei transformação logarítmica
para encolher os outliers, não enviesar o modelo'''
data.apply_log(['Valor_Condominio', 'Valor_Aluguel', 'Metragem'])

# qual a relação entre o valor do condomínio e a metragem do imóvel?
data.scattergram('Metragem', 'Valor_Condominio', "Condominio por metragem")
'''quanto maior a metragem, maior o valor do condomínio, nuvem de pontos densa e linear,
com pico de pontos em 0 e com poucos pontos espalhados da nuvem'''

# qual a relação entre o aluguel e o valor do condomínio?
data.scattergram('Valor_Condominio', 'Valor_Aluguel', "Aluguel por condominio")
'''aluguéis mais caros tendem a ter valores de condomínio mais altos, mas há dispersão lateral,
imóveis com taxas de condomínio desproporcionais, coluna de imóveis sem condomínio'''

# qual a relação entre o aluguel e a metragem do imóvel?
data.scattergram('Metragem', 'Valor_Aluguel', "Aluguel por metragem")
'''o aumento da metragem explica o aumento do aluguel, nuvem de pontos de largura relativamente constante, erro de 
previsão similar para imóveis pequenos e grandes.'''

data.heatmap()
'''
variáveis com correlação mais forte para aluguel:
    Metragem(0.73) mais forte, o tamanho do imóvel é o fator que mais explica o preço
    N_Vagas (0.66) o número de vagas pode ser um indicador de imóveis mais caros
    N_banheiros e N_Suites (0.61), mesmo peso (obs: N_banheiros e N_Suites 0.92 entre si)
'''

data.separar_base('Valor_Aluguel')

X = data.X_train[['Metragem']]  # Variável independente (características)
y = data.y_train  # Variável dependente (rótulo)
print("\nmodelo de regressão simples:")
regressao_simples = LinearRegression()
regressao_simples.fit(X, y)

#intercepto ($\beta_0$) é onde a reta cruza o eixo Y (valor quando a metragem é 0)
#coeficiente ($\beta_1$) indica quanto o aluguel aumenta para cada metro quadrado adicional
#reta: $y = \beta_1*x + \beta_0$
intercepto = regressao_simples.intercept_
coeficiente = regressao_simples.coef_[0]
print(f"Intercepto: {intercepto:.2f}")
print(f"Coeficiente: {coeficiente:.2f}")
print(f"\nEquação da Reta:")
print(f"Valor_Aluguel = {intercepto:.2f} + ({coeficiente:.2f} * Metragem)")

r2_treino = regressao_simples.score(X, y)
print(f"R² do modelo de treinamento: {r2_treino:.4f}")
'''
O R² de ~0.52 indica que a metragem sozinha explica cerca de 52% da variação do aluguel, é um resultado ok, mas mostra 
que o aluguel é influenciado por muitos outros fatores
'''

sns.scatterplot(x=X['Metragem'], y=y, alpha=0.5, label='Dados Reais')
plt.plot(X['Metragem'], regressao_simples.predict(X), color='red', linewidth=2, label='Reta de Regressão')
plt.title("Reta de Regressão: Aluguel vs Metragem")
plt.xlabel("Metragem")
plt.ylabel("Valor Aluguel")
plt.legend()
plt.show()
'''
reta passa no meio da nuvem, modelo capturou a tendência central, segue a tendência
muitos pontos distantes da reta, dispersão, explica R² não próximo de 1
dispersão dos pontos parece constante ao longo da reta, transformação logarítmica fez sentido
'''

X_test = data.X_test[['Metragem']]
y_test = data.y_test

previsoes_simples = regressao_simples.predict(X_test)
r2 = regressao_simples.score(X_test, y_test)
print("Coeficiente de Determinação (R²) nos Dados de Teste:", r2)
'''
R² de ~0.57 no teste, próximo do valor de treino, modelo não está superajustado, metragem sozinha explica apenas 
pouco mais da metade da variação do aluguel
'''

print("\nmodelo de regressão múltipla:")
regressao_multi = LinearRegression()
regressao_multi.fit(data.X_train, data.y_train)

r2_treino = regressao_multi.score(data.X_train, data.y_train)
print(f"R² do modelo de treinamento: {r2_treino:.4f}")

previsoes_multi = regressao_multi.predict(data.X_test)
r2 = regressao_multi.score(data.X_test, data.y_test)
print("Coeficiente de Determinação (R²) nos Dados de Teste:", r2)
'''
O modelo de Regressão Múltipla é mais preciso, R² de ~0.64 no teste vs ~0.57 do modelo simples,
ocorre porque o valor do aluguel tem correlação forte com outras variáveis também,
que o modelo simples ignorava, incluir elas, reduz o erro residual
'''