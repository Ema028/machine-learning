from utils.pre_processing import *

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
data.apply_log(['TSH'])
data.capping_outliers(['age', 'TT4', 'FTI'])
data.box_plot_multi(['age', 'TSH', 'TT4', 'FTI'], "Depois de Tratar Outliers")