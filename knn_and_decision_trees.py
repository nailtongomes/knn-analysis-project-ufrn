import time
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.datasets import load_breast_cancer, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

import seaborn as sns
import matplotlib.pyplot as plt

# Função para carregar e pré-processar a base Breast Cancer
def load_preprocess_breast_cancer():
    # Carrega os dados do conjunto de câncer de mama
    data = load_breast_cancer()
    X = data.data  # Características
    y = data.target  # Rótulos (maligno ou benigno)

    # Normaliza os dados para que todas as características tenham média 0 e variância 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Função para carregar e pré-processar a base Newsgroups com TF-IDF
def load_preprocess_newsgroups():
    # Carrega os dados de treino e teste sem cabeçalhos, rodapés e citações para evitar sobreajuste
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Converte o texto em vetores TF-IDF, considerando apenas as 2000 palavras mais frequentes
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)
    X_test_tfidf = vectorizer.transform(newsgroups_test.data)

    y_train = newsgroups_train.target  # Rótulos de treino
    y_test = newsgroups_test.target    # Rótulos de teste

    return X_train_tfidf, X_test_tfidf, y_train, y_test, newsgroups_train.target_names

# Função para avaliar o modelo usando várias métricas
def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)  # Realiza previsões no conjunto de teste

    # Calcula métricas de desempenho
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Exibe as métricas
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-Score:", f1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Gera a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Função para medir o tempo de treinamento e inferência do modelo
def measure_time(model, X_train, y_train, X_test):
    # Mede o tempo de treinamento
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train

    # Mede o tempo de inferência (previsão)
    start_infer = time.time()
    model.predict(X_test)
    end_infer = time.time()
    inference_time = end_infer - start_infer

    return training_time, inference_time

# Função principal para executar os experimentos na base Breast Cancer
def run_experiments_breast_cancer():
    X, y = load_preprocess_breast_cancer()  # Carrega e pré-processa os dados

    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define os hiperparâmetros para serem testados no GridSearchCV
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    dt_params = {'max_depth': [None, 5, 10, 15], 'criterion': ['gini', 'entropy']}

    # ---------- KNN ----------
    # Cria o modelo KNN
    knn = KNeighborsClassifier()
    # Realiza a busca em grade para encontrar os melhores hiperparâmetros
    grid_knn = GridSearchCV(knn, knn_params, cv=5, scoring='f1_weighted')

    # Mede o tempo de treinamento e inferência
    training_time_knn, inference_time_knn = measure_time(grid_knn, X_train, y_train, X_test)

    # Obtém o melhor modelo encontrado
    best_knn = grid_knn.best_estimator_
    print("Melhores hiperparâmetros KNN:", grid_knn.best_params_)

    # Avalia o modelo
    evaluate_model(best_knn, X_test, y_test, ['malignant', 'benign'])
    print(f"Tempo de treinamento KNN: {training_time_knn:.4f} segundos")
    print(f"Tempo de inferência KNN: {inference_time_knn:.4f} segundos")

    # ---------- Árvore de Decisão ----------
    # Cria o modelo de Árvore de Decisão
    dt = DecisionTreeClassifier(random_state=42)
    # Realiza a busca em grade para encontrar os melhores hiperparâmetros
    grid_dt = GridSearchCV(dt, dt_params, cv=5, scoring='f1_weighted')

    # Mede o tempo de treinamento e inferência
    training_time_dt, inference_time_dt = measure_time(grid_dt, X_train, y_train, X_test)

    # Obtém o melhor modelo encontrado
    best_dt = grid_dt.best_estimator_
    print("Melhores hiperparâmetros Árvore de Decisão:", grid_dt.best_params_)

    # Avalia o modelo
    evaluate_model(best_dt, X_test, y_test, ['malignant', 'benign'])
    print(f"Tempo de treinamento Árvore de Decisão: {training_time_dt:.4f} segundos")
    print(f"Tempo de inferência Árvore de Decisão: {inference_time_dt:.4f} segundos")

    # Visualização da Árvore de Decisão
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, feature_names=load_breast_cancer().feature_names, class_names=['malignant', 'benign'], filled=True)
    plt.title("Árvore de Decisão - Breast Cancer")
    plt.show()

# Função principal para executar os experimentos na base Newsgroups
def run_experiments_newsgroups():
    # Carrega e pré-processa os dados
    X_train_tfidf, X_test_tfidf, y_train, y_test, target_names = load_preprocess_newsgroups()

    # Define os hiperparâmetros para serem testados no GridSearchCV
    knn_params = {'n_neighbors': [3, 5, 7]}
    dt_params = {'max_depth': [None, 10, 20], 'criterion': ['gini', 'entropy']}

    # ---------- KNN ----------
    # Cria o modelo KNN
    knn = KNeighborsClassifier()
    # Realiza a busca em grade
    grid_knn = GridSearchCV(knn, knn_params, cv=3, scoring='f1_weighted')

    # Mede o tempo de treinamento e inferência
    training_time_knn, inference_time_knn = measure_time(grid_knn, X_train_tfidf, y_train, X_test_tfidf)

    # Obtém o melhor modelo encontrado
    best_knn = grid_knn.best_estimator_
    print("Melhores hiperparâmetros KNN:", grid_knn.best_params_)

    # Avalia o modelo
    evaluate_model(best_knn, X_test_tfidf, y_test, target_names)
    print(f"Tempo de treinamento KNN: {training_time_knn:.4f} segundos")
    print(f"Tempo de inferência KNN: {inference_time_knn:.4f} segundos")

    # ---------- Árvore de Decisão ----------
    # Cria o modelo de Árvore de Decisão
    dt = DecisionTreeClassifier(random_state=42)
    # Realiza a busca em grade
    grid_dt = GridSearchCV(dt, dt_params, cv=3, scoring='f1_weighted')

    # Mede o tempo de treinamento e inferência
    training_time_dt, inference_time_dt = measure_time(grid_dt, X_train_tfidf, y_train, X_test_tfidf)

    # Obtém o melhor modelo encontrado
    best_dt = grid_dt.best_estimator_
    print("Melhores hiperparâmetros Árvore de Decisão:", grid_dt.best_params_)

    # Avalia o modelo
    evaluate_model(best_dt, X_test_tfidf, y_test, target_names)
    print(f"Tempo de treinamento Árvore de Decisão: {training_time_dt:.4f} segundos")
    print(f"Tempo de inferência Árvore de Decisão: {inference_time_dt:.4f} segundos")

    # Visualização da Árvore de Decisão (pode ser muito grande)
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, max_depth=3, filled=True)
    plt.title("Árvore de Decisão (Profundidade Limitada) - Newsgroups")
    plt.show()

# Função para experimentar diferentes hiperparâmetros e analisar seu impacto
def hyperparameter_analysis(X_train, y_train, X_test, y_test, model_type='knn'):
    if model_type == 'knn':
        # Lista de valores de k para testar
        neighbors = list(range(1, 11))
        f1_scores = []
        for k in neighbors:
            # Cria e treina o modelo KNN com k vizinhos
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            # Calcula o F1-Score
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)
        # Plota o impacto de k no F1-Score
        plt.figure(figsize=(10, 6))
        plt.plot(neighbors, f1_scores, marker='o')
        plt.title('Impacto do número de vizinhos no F1-Score (KNN)')
        plt.xlabel('Número de Vizinhos (k)')
        plt.ylabel('F1-Score')
        plt.show()
    elif model_type == 'dt':
        # Lista de profundidades máximas para testar
        depths = [None, 5, 10, 15, 20, 25]
        f1_scores = []
        for depth in depths:
            # Cria e treina o modelo de Árvore de Decisão com profundidade máxima especificada
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            # Calcula o F1-Score
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)
        # Plota o impacto da profundidade máxima no F1-Score
        plt.figure(figsize=(10, 6))
        plt.plot([str(d) for d in depths], f1_scores, marker='o')
        plt.title('Impacto da profundidade máxima no F1-Score (Árvore de Decisão)')
        plt.xlabel('Profundidade Máxima')
        plt.ylabel('F1-Score')
        plt.show()

# Executando os experimentos
if __name__ == '__main__':
    print("----- Experimentos na base Breast Cancer -----")
    run_experiments_breast_cancer()

    print("\n\n----- Experimentos na base Newsgroups -----")
    run_experiments_newsgroups()

    # Análise de hiperparâmetros para Breast Cancer
    X_bc, y_bc = load_preprocess_breast_cancer()
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.3, random_state=42)
    print("\nAnálise de Hiperparâmetros - Breast Cancer - KNN")
    hyperparameter_analysis(X_train_bc, y_train_bc, X_test_bc, y_test_bc, model_type='knn')
    print("\nAnálise de Hiperparâmetros - Breast Cancer - Árvore de Decisão")
    hyperparameter_analysis(X_train_bc, y_train_bc, X_test_bc, y_test_bc, model_type='dt')

    # Análise de hiperparâmetros para Newsgroups
    X_train_ng, X_test_ng, y_train_ng, y_test_ng, target_names_ng = load_preprocess_newsgroups()
    print("\nAnálise de Hiperparâmetros - Newsgroups - KNN")
    hyperparameter_analysis(X_train_ng, y_train_ng, X_test_ng, y_test_ng, model_type='knn')
    print("\nAnálise de Hiperparâmetros - Newsgroups - Árvore de Decisão")
    hyperparameter_analysis(X_train_ng, y_train_ng, X_test_ng, y_test_ng, model_type='dt')
