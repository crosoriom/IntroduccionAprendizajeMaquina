from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
import pandas as pd
from time import time
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv('database.csv')

x_train, x_test = train_test_split(df, test_size = 0.2)
output = 'Total_Points'
y_train, y_test = x_train[output], x_test[output]
x_train.drop(columns = output, inplace = True)
x_test.drop(columns = output, inplace = True)

models = [
    ('LinearRegression', LinearRegression(),
     {}),                                               #La regresión lineal no admite hiperparámetros

    ('Lasso', Lasso(),
     {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),

    ('ElasticNet', ElasticNet(),
     {'alpha': [0.001, 0.01, 0.1, 1, 10],
      'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]}),

    ('Ridge', Ridge(),
     {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),

    ('KernelRidge', KernelRidge(),
     {'alpha': [0.01, 0.1, 1],
      'kernel': ['rbf', 'linear', 'poly'],
      'gamma': [1, 0.1, 0.01]}),

    ('SGDRegressor', SGDRegressor(),
     {'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
      'loss': ['squared_error', 'huber', 'epsilon_insensitive']}),

    ('BayesianRidge', BayesianRidge(),
     {'alpha_1': [1e-6, 1e-4, 1e-2],
      'alpha_2': [1e-6, 1e-4, 1e-2],
      'lambda_1': [1e-6, 1e-4, 1e-2],
      'lambda_2': [1e-6, 1e-4, 1e-2]}),

    ('GaussianProcess', GaussianProcessRegressor(),
     {'kernel': [RBF(1.0),
                  Matern(length_scale = 1.0, nu = 1.5),
                  C(1.0) * RBF(1.0),
                  RBF(1.0) * WhiteKernel(noise_level = 1.0)],
      'alpha': [1e-10, 1e-7, 1e-5],
      'n_restarts_optimizer': [0, 1, 3]}),

    ('RandomForest', RandomForestRegressor(),
     {'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20],
      'min_samples_split': [5, 10, 20]}),

    ('SVR', LinearSVR(),
     {'C': [0.1, 1, 10, 100],
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']})
]

results = []

for name, model, params in models:
    print(f'\nEvaluando {name}...')
    start = time()
    
    # Corregir los nombres de las métricas y añadir refit
    gridSearch = GridSearchCV(
        estimator = model,
        param_grid = params,
        cv = 5,
        scoring = {
            'mse': 'neg_mean_squared_error', 
            'mae': 'neg_mean_absolute_error'
        },
        refit = 'mse',  # Especificar qué métrica usar para seleccionar el mejor modelo
        n_jobs = -1,
        verbose = 1
    )
    
    gridSearch.fit(x_train, y_train)
    trainingTime = time() - start
    
    predictions = gridSearch.predict(x_test)
    
    # Cálculo de métricas en el conjunto de test
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    model_results = {
        'Model': name,
        'Best_Params': gridSearch.best_params_,
        'Best_MSE_CV': -gridSearch.cv_results_['mean_test_mse'][gridSearch.best_index_],
        'Best_MAE_CV': -gridSearch.cv_results_['mean_test_mae'][gridSearch.best_index_],
        'Test_MAE': mae,
        'Test_MSE': mse,
        'Test_R2': r2,
        'Test_MAPE': mape,
        'Training_Time': trainingTime
    }
    
    results.append(model_results)
    
    print(f"{name} - Mejor MSE CV: {-gridSearch.cv_results_['mean_test_mse'][gridSearch.best_index_]:.4f}, Test MSE: {mse:.4f}, R²: {r2:.4f}")
    print(f"Mejores parámetros: {gridSearch.best_params_}")
    print(f"Tiempo de entrenamiento: {trainingTime:.2f} segundos")

resultsDF = pd.DataFrame(results)
resultsDF.to_csv('Model comparison results.csv', index = False)
