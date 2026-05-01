from utils.pre_processing import *

'''
o objetivo desse modelo é diagnosticar e classificar pacientes
para doenças da tireoide com base em perfis demográficos e resultados de exames 
clínicos (TSH, T3, TT4, FTI). 
'''
df = pd.read_csv("../data/hypothyroid.csv", delimiter=',')
data = Dataframe(df)
print(data.df.info()) #só categóricas

data.to_number(['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'])
data.print_missing()
data.drop_columns(['TBG', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured',
                   'FTI measured', 'TBG measured']) #TBG falta tudo, não traz informação e colunas de measured só diz se foi feito exame
data.df.dropna(subset=['age'], inplace=True) #só 1 linha sem idade, remover ela
data.imputar_knn(['T3', 'T4U', 'FTI', 'TSH', 'TT4'])  #usa média de 5 pacientes parecidos

data.print_unique_values()
#remapeando booleanas
colunas_booleanas = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                     'sick', 'pregnant', 'thyroid surgery', 'I131 treatment',
                     'query hypothyroid', 'query hyperthyroid', 'lithium',
                     'goitre', 'tumor', 'hypopituitary', 'psych']
data.df[colunas_booleanas] = data.df[colunas_booleanas].replace({'f': 0, 't': 1})
data.df['binaryClass'] = data.df['binaryClass'].replace({'N': 0, 'P': 1})
data.df['sex'] = data.df['sex'].replace({'?': np.nan, 'F': 0, 'M': 1}) #150 faltando(~4%)
data.df['sex'] = data.df['sex'].fillna(df['sex'].mode()[0])

print(data.df.describe().T) #sinais de outlier em 'age', 'TSH', 'TT4', 'FTI'
data.box_plot_multi(['age', 'TSH', 'TT4', 'FTI'], "Antes de Tratar Outliers")
data.apply_log(['TSH', 'TT4', 'FTI']) #cauda longa bilateral, log para melhorar escala
data.capping_outliers(['age']) #só 1 valor mt distoante(vampiro de 455 anos)
data.box_plot_multi(['TSH', 'TT4', 'FTI'], "Depois de Tratar Outliers")
'''melhor análise visual, caudas longas ainda lá, mas capping evitado para não descaracterizar os exames
o q faz sentido pela escolha do algoritmo XGBoost que não é tão sensível a outliers e 
não precisam de escalonamento(por isso std scaler não usada)'''

#print(data.df.info())
dicionario_encoder = data.label_encoding()
#data.print_missing() #um paciente com todos os exames nulos imputar_knn não conseguiu substituir o valor
data.df = data.df.dropna()
data.heatmap()
'''maioria tem pouca correlação com referral source, 
psych tem uma relação negativa de -0.40 sendo a que mais influencia'''
data.separar_base('referral source')
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'referral source')
print(f"Análise de Multicolinearidade\n: {data.get_vif()}\n") #TT4, FTI e T4U têm alta redundância