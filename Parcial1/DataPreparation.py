import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
from RenameDatabase import renameDatabase

warnings.filterwarnings(action = 'ignore', message = '^internal gelsd')

mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

try:
    os.mkdir('results')
except:
    print("Carpeta results ya existe")

df = pd.read_csv('2023_nba_player_stats.csv')
scorePrediction = renameDatabase(df)

# Elimincación de características irrelevantes o redundantes y filas con valores nulos
columns_drop = ['Player_Name', 'Team', 'Position','Field_Goals_Made', 'Field_Goal_Percentage', 'Three_Point_FG_Made', 'Three_Point_FG_Percentage', 'Free_Throws_Made', 'Free_Throws_Percentage', 'Wins', 'Loses', 'Defensive_Rebounds', 'Offensive_Rebounds', 'Blocks', 'Double_Doubles', 'Triple_Doubles']
scorePrediction.drop(columns = columns_drop, inplace = True)
scorePrediction.dropna(inplace = True)

correlationMatrix = scorePrediction.corr()
#print('Matriz de correlación:')
#print(correlationMatrix)
plt.figure(figsize = (16, 12))
sns.heatmap(correlationMatrix, annot = False)
plt.title('Correlation Matrix')
plt.savefig('results/correlation_matrix.png')

pd.plotting.scatter_matrix(scorePrediction, figsize = (16, 12))
plt.tight_layout()
plt.savefig('results/scatter_matrix')

# Normalizamos las edades para graficarlas con respecto al puntaje para ver si hay alguna relación
plt.figure()
points_per_age = scorePrediction.groupby('Age')['Total_Points'].mean()
plt.title('points_per_age')
plt.xlabel('Age')
plt.ylabel('Mean Points')
plt.savefig('results/points_per_age.png')
scorePrediction.drop(columns = 'Age', inplace = True)

new_df = scorePrediction
new_df.to_csv('database.csv', index = False)
