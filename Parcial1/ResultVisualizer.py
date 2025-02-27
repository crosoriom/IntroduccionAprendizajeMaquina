import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('Model comparison results.csv')

plt.figure(figsize=(14, 10))
# 1. Gráfico de barras para MSE
plt.subplot(2, 2, 1)
results = results_df.sort_values('Test_MSE')
ax = sns.barplot(x='Model', y='Test_MSE', data=results)
plt.xticks(rotation=45, ha='right')
plt.title('Error Cuadrático Medio (MSE) por Modelo', fontsize=12)
plt.tight_layout()

# 2. Gráfico de barras para R²
plt.subplot(2, 2, 2)
results = results_df.sort_values('Test_R2', ascending=False)
ax = sns.barplot(x='Model', y='Test_R2', data=results)
plt.xticks(rotation=45, ha='right')
plt.title('Coeficiente de Determinación (R²) por Modelo', fontsize=12)
plt.ylim(0, 1.05*max(results['Test_R2']))
plt.tight_layout()

# 3. Gráfico de barras para MAE
plt.subplot(2, 2, 3)
results = results_df.sort_values('Test_MAE')
ax = sns.barplot(x='Model', y='Test_MAE', data=results)
plt.xticks(rotation=45, ha='right')
plt.title('Error Absoluto Medio (MAE) por Modelo', fontsize=12)
plt.tight_layout()

# 4. Comparación de tiempo de entrenamiento
plt.subplot(2, 2, 4)
results = results_df.sort_values('Training_Time')
ax = sns.barplot(x='Model', y='Training_Time', data=results)
plt.xticks(rotation=45, ha='right')
plt.title('Tiempo de Entrenamiento (segundos) por Modelo', fontsize=12)
plt.tight_layout()

plt.tight_layout(pad=3.0)
plt.savefig('results/model_comparison.png')
plt.close()

# 5. Gráfico de dispersión: R² vs MSE (con tamaño = tiempo de entrenamiento)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    x='Test_MSE', 
    y='Test_R2', 
    s=results_df['Training_Time']*10,  # Tamaño proporcional al tiempo
    alpha=0.7,
    c=range(len(results_df)),  # Colorear por índice
    cmap='viridis',
    data=results_df
)

# Añadir etiquetas a cada punto
for i, model in enumerate(results_df['Model']):
    plt.annotate(
        model,
        (results_df['Test_MSE'].iloc[i], results_df['Test_R2'].iloc[i]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9
    )

plt.title('Comparación de Modelos: R² vs MSE', fontsize=14)
plt.xlabel('Error Cuadrático Medio (MSE)')
plt.ylabel('Coeficiente de Determinación (R²)')
plt.grid(True, linestyle='--', alpha=0.7)
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4, 
                                         fmt="{x:.1f}s", func=lambda x: x/10)
plt.legend(handles, labels, loc="lower right", title="Tiempo de\nEntrenamiento")
plt.savefig('results/model_scatter.png')
