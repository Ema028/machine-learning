from utils.pre_processing import *

df = pd.read_csv("../data/campaign.csv", delimiter=';')
print(df.info()) #tipos certos, 2 categóricas
data = Dataframe(df)

data.print_missing() #income com 1.071429% de valores faltando(24), pouco expressivo
data.drop_missing()

data.print_unique_values()
status_mapping = {
    'Married': 'Partner',
    'Together': 'Partner',
    'Divorced': 'Single',
    'Widow': 'Single',
    'Alone': 'Single',
    'Absurd': 'Single',
    'YOLO': 'Single'
}
data.df['Marital_Status'] = df['Marital_Status'].replace(status_mapping)

education_mapping = {
    'Basic': 'Undergraduate',
    'Graduation': 'Graduate',
    'Master': 'Postgraduate',
    '2n Cycle': 'Postgraduate',
    'PhD': 'Postgraduate'
}
data.df['Education'] = df['Education'].replace(education_mapping)

print(data.df.describe().T) #colunas de gastos com cauda longa
data.box_plot_multi(['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],"Gastos Anuais")
data.apply_log(['MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])

data.box_plot(None, 'Income', "Boxplot de Renda Anual") #ponto muito longe do resto
data.box_plot(None, 'Year_Birth', "Boxplot de nascimento") #poucas datas muito antigas(seres imortais)
data.capping_outliers(['Year_Birth', 'Income'])

data.heatmap()
data.one_hot()
data.one_hot_heatmap()

#funil de conversão funciona?
data.bar_plot('NumWebVisitsMonth', 'WebPurchases', "Compras por Visitas no Site")
'''pico de conversão com duas visitas mensais, alta intenção de compra
quarta visita pra frente, a taxa caí progressivamente e de treze pra frente zera, repique em nove e dez visitas, provavelmente caçadores de promoções
maior chance de fechar a venda nas três primeiras interações'''

#concentração de clientes
data.bar_plot('NumStorePurchases', 'WebPurchases', 'Compras em Loja Física vs Web')
'''não tem canibalização, clientes com até três compras na loja física pouca conversão online,
a partir da quarta ida à loja taxa de conversão alta, cliente converte-se naturalmente em um comprador digital'''

data.separar_base('WebPurchases')
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'WebPurchases') #balanceado já
data.std_scaler()
