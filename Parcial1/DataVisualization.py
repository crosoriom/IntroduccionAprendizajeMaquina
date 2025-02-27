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
rows, columns = df.shape
print(f'This DataSet has {rows} rows and {columns} columns')
print('Number of duplicated data: ', df.duplicated().sum())

scorePrediction = renameDatabase(df)

scorePrediction.hist(bins = 50, figsize = (20, 15))
plt.tight_layout()
plt.savefig('results/attribute_histogram_plots.png')

copy = scorePrediction
features = copy.drop(columns = ['Total_Points', 'Player_Name', 'Team', 'Position'])
target = scorePrediction['Total_Points']

plt.figure(figsize = (25, 20))
for i, column in enumerate(features.columns):
    plt.subplot(6, 5, i + 1)
    sns.scatterplot(x = features[column], y = target, hue = scorePrediction['Position'])
    plt.title(f'{column} vs Total_Points')
    plt.xlabel(column)
    plt.ylabel('Total_Points')
plt.tight_layout()
plt.savefig('results/points_scater_diagrams.png')

plt.figure()
sns.scatterplot(x = scorePrediction['Field_Goals_Attempted'], y = target, hue = scorePrediction['Age'])
plt.xlabel('Field_Goals_Attempted')
plt.ylabel('Total_Points')
plt.savefig('results/age_goals_distribution.png')

head = scorePrediction.head()
head.to_markdown('results/Table_Head.md')
descriptionTable = scorePrediction.describe()
descriptionTable.to_markdown('results/Description_Table.md', index = False)
scorePrediction.info()
