# pip install -r requirements.txt
# Autor: Nailton Gomes Silva - @nailtongs

import io
import sys

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


class MeuKNN:
    """
    Classe para implementação do KNN do zero

    Pontos próximos no espaço das características tendem a pertencer à mesma classe
    """

    def __init__(self, k=3):
        # número de vizinhos mais próximos que o
        # modelo considerará ao fazer uma previsão
        self.k = k

        # O valor de k é crítico; um k muito pequeno pode tornar o modelo sensível a ruídos,
        # enquanto um k muito grande pode diluir a influência dos vizinhos mais próximos.

    def fit(self, X_train, y_train):
        # Matriz contendo as características (features) dos dados de treinamento
        self.X_train = X_train
        # Vetor contendo os rótulos (classes) correspondentes aos dados de treinamento.
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calcula a distância euclidiana entre dois pontos x1 e x2
        # Calcula a distância euclidiana entre x e todos os exemplos de treinamento
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Obter os índices dos k vizinhos mais próximos
        # Array contendo os índices dos k menores valores em distances, ou seja,
        # os índices dos k pontos de treinamento mais próximos
        k_indices = np.argsort(distances)[:self.k]
        # Obter as classes desses vizinhos
        # Rótulos dos k Vizinhos Mais Próximos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Votar na classe mais comum
        # classe que aparece com mais frequência.
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def redirect_output(func):

    """
    Redireciona a saída padrão para um arquivo de texto
    """

    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        output_file = io.StringIO()
        sys.stdout = output_file

        result = func(*args, **kwargs)

        sys.stdout = original_stdout

        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(output_file.getvalue())

        return result

    return wrapper


# Dados de treinamento
def create_data():

    # Dados sintéticos 2D
    X_synthetic, y_synthetic = make_classification(
        n_samples=1000, n_features=2, n_redundant=0,
        n_clusters_per_class=1, n_classes=3
    )
    # Base Iris
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    # Base Wine
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    return X_synthetic, y_synthetic, X_iris, y_iris, X_wine, y_wine


def preprocess_data(X, y, train_size):
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test


def run_experiments(X, y, dataset_name='t'):
    results = []

    for train_size in [0.1, 0.2, 0.3, 0.4]:

        for k in range(1, 10):

            X_train, X_test, y_train, y_test = preprocess_data(X, y, train_size)

            # KNN do zero
            knn_custom = MeuKNN(k)  # n_neighbors
            knn_custom.fit(X_train, y_train)
            y_pred_custom = knn_custom.predict(X_test)

            # Inicialize o classificador com k vizinhos
            classifier = KNeighborsClassifier(k)
            classifier.fit(X_train, y_train)
            y_pred_sklearn = classifier.predict(X_test)

            # Métricas
            for model_name, y_pred in [("Custom", y_pred_custom), ("Scikit-learn", y_pred_sklearn)]:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                results.append({
                    'Dataset': dataset_name,
                    'Train Size': train_size,
                    'k': k,
                    'Model': model_name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1,
                    'Confusion Matrix': cm
                })

    return results


def get_results():

    X_synthetic, y_synthetic, X_iris, y_iris, X_wine, y_wine = create_data()

    results_synthetic = run_experiments(X_synthetic, y_synthetic, 'Synthetic')
    results_iris = run_experiments(X_iris, y_iris, 'Iris')
    results_wine = run_experiments(X_wine, y_wine, 'Wine')

    return results_synthetic, results_iris, results_wine


def plot_results(results, dataset_name):
    df = pd.DataFrame(results)
    df = df[df['Dataset'] == dataset_name]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for train_size in df['Train Size'].unique():
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        for ax, metric in zip(axs.flatten(), metrics):
            sns.lineplot(data=df[df['Train Size'] == train_size], x='k', y=metric, hue='Model', marker='o', ax=ax)
            ax.set_title(f'{metric} vs k')
            ax.set_xlabel('Número de Vizinhos (k)')
            ax.set_ylabel(metric)
            ax.legend(title='Modelo')
        fig.suptitle(f'{dataset_name} Dataset - Tamanho de Treino: {int(train_size*100)}%')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(f'./images/{dataset_name}_train_size_{int(train_size*100)}_metrics.png')


# Métricas vs Tamanho de Treinamento para o melhor k:
def plot_metrics_vs_train_size_best_k(results, dataset_name, best_k):
    df = pd.DataFrame(results)
    df = df[(df['Dataset'] == dataset_name) & (df['k'] == best_k)]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, metric in zip(axs.flatten(), metrics):
        sns.lineplot(data=df, x='Train Size', y=metric, hue='Model', marker='o', ax=ax)
        ax.set_title(f'{metric} vs Tamanho de Treino (k={best_k})')
        ax.set_xlabel('Tamanho de Treino')
        ax.set_ylabel(metric)
        ax.legend(title='Modelo')
    fig.suptitle(f'{dataset_name} Dataset - k = {best_k}')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(f'./images/{dataset_name}_k_{best_k}_metrics_vs_train_size.png')


# Plotagem das Matrizes de Confusão para o melhor desempenho:
def plot_confusion_matrices(results, dataset_name):
    df = pd.DataFrame(results)
    df = df[df['Dataset'] == dataset_name]
    models = df['Model'].unique()
    for model in models:
        df_model = df[df['Model'] == model]
        best_row = df_model.loc[df_model['F1-Score'].idxmax()]
        cm = best_row['Confusion Matrix']
        k = best_row['k']
        train_size = best_row['Train Size']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {dataset_name} - {model}\nk={k}, Tamanho de Treino={int(train_size*100)}%')
        plt.ylabel('Rótulo Verdadeiro')
        plt.xlabel('Rótulo Predito')
        plt.tight_layout()
        plt.savefig(f'./images/{dataset_name}_{model}_best_confusion_matrix.png')


# Plotagem de Heatmap do F1-Score em função de k e Tamanho de Treino
def plot_heatmap_f1_score(results, dataset_name):
    df = pd.DataFrame(results)
    df = df[df['Dataset'] == dataset_name]
    models = df['Model'].unique()
    for model in models:
        df_model = df[df['Model'] == model]
        pivot_table = df_model.pivot_table(values='F1-Score', index='Train Size', columns='k')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'Heatmap F1-Score - {dataset_name} - {model}')
        plt.xlabel('k')
        plt.ylabel('Tamanho de Treino')
        plt.tight_layout()
        plt.savefig(f'./images/{dataset_name}_{model}_f1_score_heatmap.png')


def plot_overall_model_performance(results):
    df = pd.DataFrame(results)
    avg_f1_scores = df.groupby(['Dataset', 'Model'])['F1-Score'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_f1_scores, x='Dataset', y='F1-Score', hue='Model')
    plt.title('F1-Score Médio por Modelo nos Conjuntos de Dados')
    plt.xlabel('Conjunto de Dados')
    plt.ylabel('F1-Score Médio')
    plt.legend(title='Modelo')
    plt.tight_layout()
    plt.savefig('./images/overall_model_performance.png')


def analyze_results(results):

    df = pd.DataFrame(results)

    # Lista para armazenar os resumos por conjunto de dados
    summaries = []

    # Iterar sobre cada conjunto de dados
    for dataset in df['Dataset'].unique():
        df_dataset = df[df['Dataset'] == dataset]

        # Encontrar o modelo com melhor desempenho médio
        mean_f1 = df_dataset.groupby('Model')['F1-Score'].mean()
        best_model = mean_f1.idxmax()

        # Analisar o impacto de k na performance
        mean_f1_by_k = df_dataset.groupby('k')['F1-Score'].mean()
        best_k = mean_f1_by_k.idxmax()

        # Analisar o impacto do tamanho do conjunto de treinamento
        mean_f1_by_train_size = df_dataset.groupby('Train Size')['F1-Score'].mean()
        best_train_size = mean_f1_by_train_size.idxmax()

        # Adicionar ao resumo
        summaries.append({
            'Dataset': dataset,
            'Best Model': best_model,
            'Best k': best_k,
            'Best Train Size': best_train_size,
            'Model Performance': mean_f1.to_dict(),
            'k Performance': mean_f1_by_k.to_dict(),
            'Train Size Performance': mean_f1_by_train_size.to_dict()
        })

    return summaries


def identify_best_model(results):
    df = pd.DataFrame(results)
    datasets = df['Dataset'].unique()
    best_models = {}

    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        avg_f1_scores = dataset_df.groupby('Model')['F1-Score'].mean()
        best_model = avg_f1_scores.idxmax()
        best_models[dataset] = best_model
        print(f"No conjunto de dados '{dataset}', o modelo com melhor desempenho médio é: {best_model}")

    return best_models


def compare_models_overall(results):
    df = pd.DataFrame(results)
    avg_f1_scores = df.groupby('Model')['F1-Score'].mean()
    print("Desempenho Médio Geral dos Modelos:")
    print(avg_f1_scores)

    better_model = avg_f1_scores.idxmax()
    print(f"\nO modelo com melhor desempenho médio geral é: {better_model}")

    return better_model


def best_k_per_dataset(results):
    df = pd.DataFrame(results)
    datasets = df['Dataset'].unique()
    best_k_values = {}

    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        avg_f1_scores = dataset_df.groupby('k')['F1-Score'].mean()
        best_k = avg_f1_scores.idxmax()
        best_k_values[dataset] = best_k
        print(f"Para o conjunto de dados '{dataset}', o melhor desempenho médio foi alcançado com k={best_k}.")

    return best_k_values


def analyze_training_percentage(results):
    df = pd.DataFrame(results)
    avg_f1_scores = df.groupby('Train Size')['F1-Score'].mean()
    print("Desempenho Médio por Percentual de Treinamento:")
    print(avg_f1_scores.sort_index())

    best_train_size = avg_f1_scores.idxmax()
    print(f"\nO melhor desempenho médio foi obtido com {int(best_train_size * 100)}% de dados de treinamento.")

    return best_train_size


def analyze_correlation(results):
    df = pd.DataFrame(results)
    df['Train Size Percentage'] = df['Train Size'] * 100
    correlation_k = df['k'].corr(df['F1-Score'])
    correlation_train_size = df['Train Size'].corr(df['F1-Score'])

    print(f"Correlação entre k e F1-Score: {correlation_k:.4f}")
    print(f"Correlação entre Percentual de Treinamento e F1-Score: {correlation_train_size:.4f}")

    return correlation_k, correlation_train_size


@redirect_output
def run():

    results_synthetic, results_iris, results_wine = get_results()
    plot_results(results_synthetic, 'Synthetic')
    plot_results(results_iris, 'Iris')
    plot_results(results_wine, 'Wine')

    plot_confusion_matrices(results_synthetic, 'Synthetic')
    plot_confusion_matrices(results_iris, 'Iris')
    plot_confusion_matrices(results_wine, 'Wine')

    plot_heatmap_f1_score(results_synthetic, 'Synthetic')
    plot_heatmap_f1_score(results_iris, 'Iris')
    plot_heatmap_f1_score(results_wine, 'Wine')

    plot_overall_model_performance(results_synthetic + results_iris + results_wine)

    # Unificando todos os resultados
    all_results = results_synthetic + results_iris + results_wine

    # Analisando os resultados
    summaries = analyze_results(all_results)

    # Exibindo os resumos
    for summary in summaries:
        print(f"Dataset: {summary['Dataset']}")
        print(f"Melhor Modelo: {summary['Best Model']}")
        print(f"Melhor k: {summary['Best k']}")
        print(f"Melhor Percentual de Treinamento: {int(summary['Best Train Size']*100)}%")
        print("\nDesempenho Médio por Modelo (F1-Score):")
        for model, score in summary['Model Performance'].items():
            print(f"  {model}: {score:.4f}")
        print("\nDesempenho Médio por k (F1-Score):")
        for k_value, score in summary['k Performance'].items():
            print(f"  k={k_value}: {score:.4f}")
        print("\nDesempenho Médio por Percentual de Treinamento (F1-Score):")
        for train_size, score in summary['Train Size Performance'].items():
            print(f"  {int(train_size*100)}%: {score:.4f}")
        print("-" * 50)

    best_models = identify_best_model(results_synthetic + results_iris + results_wine)
    overall_best_model = compare_models_overall(results_synthetic + results_iris + results_wine)
    best_k_values = best_k_per_dataset(results_synthetic + results_iris + results_wine)
    best_train_size = analyze_training_percentage(results_synthetic + results_iris + results_wine)
    correlation_k, correlation_train_size = analyze_correlation(results_synthetic + results_iris + results_wine)

    # Plotagem das métricas vs Tamanho de Treino para o melhor k
    for dataset in ['Synthetic', 'Iris', 'Wine']:
        plot_metrics_vs_train_size_best_k(all_results, dataset, best_k_values[dataset])

    final_results = {
        'best_models': best_models,
        'overall_best_model': overall_best_model,
        'best_k_values': best_k_values,
        'best_train_size': best_train_size,
        'correlation_k': correlation_k,
        'correlation_train_size': correlation_train_size
    }

    return final_results


if __name__ == '__main__':
    run()
