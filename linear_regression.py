from utils.pre_processing import *

df = pd.read_csv("data/aluguel.csv", delimiter=';')
print(df.info())  #apenas variáveis numéricas
data = Dataframe(df)
data.print_missing() #sem valor faltando
print(df.describe().T)
'''Valor_Aluguel, Valor_Condominio e Metragem, talvez outliers, pois Q3 muito diferente do valor máximo
variáveis quantitativas (quartos, banheiros, suítes e vagas) com máximos altos, 
mas faz sentido para imóveis de alto padrão'''

sns.histplot(df['Valor_Aluguel'], bins=30)
plt.title("Distribuição de preço do aluguel")
plt.show()
#Valor_Aluguel com valores extremos muito acentuados, cauda muito longa, muita assimetria

sns.histplot(df['Valor_Condominio'], bins=30)
plt.title("Distribuição de preço do condomínio")
plt.show()
#cauda longa também, menos que aluguel, bastante disperção

sns.histplot(df['Metragem'], bins=30)
plt.title("Distribuição de metragem do imóvel")
plt.show()
#cauda longa também, distorção, mas menos que as outras

'''tentei tratamento com capping pelo IQR, mas fez um teto artificial e para não remover dados usei transformação logarítmica
para encolher os outliers, não enviesar o modelo'''
data.apply_log(['Valor_Condominio', 'Valor_Aluguel', 'Metragem'])

# qual a relação entre o valor do condomínio e a metragem do imóvel?
sns.scatterplot(x='Metragem', y='Valor_Condominio', data=df)
plt.title("Condominio por metragem")
plt.xlabel("Metragem")
plt.ylabel("Valor_Condominio")
plt.show()
'''quanto maior a metragem, maior o valor do condomínio, nuvem de pontos densa e linear,
com pico de pontos em 0 e com poucos pontos espalhados da nuvem'''

# qual a relação entre o aluguel e o valor do condomínio?
sns.scatterplot(x='Valor_Condominio', y='Valor_Aluguel', data=df)
plt.title("Aluguel por condominio")
plt.xlabel("Valor_Condominio")
plt.ylabel("Valor_Aluguel")
plt.show()
'''aluguéis mais caros tendem a ter valores de condomínio mais altos, mas há dispersão lateral,
imóveis com taxas de condomínio desproporcionais, coluna de imóveis sem condomínio'''

# qual a relação entre o aluguel e a metragem do imóvel?
sns.scatterplot(x='Metragem', y='Valor_Aluguel', data=df)
plt.title("Aluguel por metragem")
plt.xlabel("Metragem")
plt.ylabel("Valor_Aluguel")
plt.show()
'''o aumento da metragem explica o aumento do aluguel, nuvem de pontos de largura relativamente constante, erro de 
previsão similar para imóveis pequenos e grandes.'''
