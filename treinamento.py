# MÓDULO DE TREINAMENTO - Classificador Breast Cancer (atributo Class)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV  # hiperparametrizador
from sklearn.metrics import accuracy_score  # acurácia do modelo
from sklearn.tree import DecisionTreeClassifier  # algoritmo classificador
from imblearn.over_sampling import SMOTE  # para balancear os dados
from pickle import dump

# 1) CARREGAR OS DADOS
dados = pd.read_csv('breast-cancer.csv')
print('Formato da base:', dados.shape)

# 2) TRATAR VALORES FALTANTES
# Na base, os valores desconhecidos estão marcados com '?'
# (node-caps tem 8 e breast-quad tem 1). Eles são trocados pela moda da coluna.
dados = dados.replace('?', np.nan)
print('\nValores faltantes:')
print(dados.isna().sum())
for coluna in ['node-caps', 'breast-quad']:
    dados[coluna] = dados[coluna].fillna(dados[coluna].mode()[0])

# 3) SEPARAR ATRIBUTOS E CLASSE
dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']
print('\nQuantidade por classe:')
print(dados_classes.value_counts())

# 4) TRANSFORMAR OS ATRIBUTOS CATEGÓRICOS EM NÚMEROS
# Atributos em faixas (age, tumor-size, inv-nodes) têm ordem, então cada faixa
# vira o seu valor do meio. Exemplo: '50-59' -> 54.5
colunas_faixas = ['age', 'tumor-size', 'inv-nodes']


def faixa_para_numero(faixa):
    inicio, fim = faixa.split('-')
    return (int(inicio) + int(fim)) / 2


for coluna in colunas_faixas:
    dados_atributos[coluna] = dados_atributos[coluna].apply(faixa_para_numero)

# Atributos sem ordem (nominais) viram colunas dummy (0 ou 1)
colunas_nominais = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']
dados_atributos = pd.get_dummies(dados_atributos, columns=colunas_nominais, dtype=int)

# Atributos numéricos (são os que serão normalizados)
colunas_numericas = colunas_faixas + ['deg-malig']
print('\nAtributos usados pelo modelo:', list(dados_atributos.columns))

# 5) SEPARAR DADOS DE TREINO (70%) E DE TESTE (30%)
# A acurácia é medida em pacientes que o modelo não viu no treinamento
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, stratify=dados_classes, random_state=42)

# 6) BALANCEAR OS DADOS DE TREINO
resampler = SMOTE(random_state=42)
atributos_treino_bal, classes_treino_bal = resampler.fit_resample(atributos_treino, classes_treino)
print('\nTreino depois do balanceamento:')
print(classes_treino_bal.value_counts())

# 7) NORMALIZAR OS ATRIBUTOS NUMÉRICOS
# Somente os atributos são normalizados (a classe não).
# As colunas dummy já estão entre 0 e 1, então não passam pelo normalizador.
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(atributos_treino_bal[colunas_numericas])
with open('normalizador_breast.model', 'wb') as arquivo:
    dump(modelo_normalizador, arquivo)

atributos_treino_norm = atributos_treino_bal.copy()
atributos_treino_norm[colunas_numericas] = modelo_normalizador.transform(
    atributos_treino_bal[colunas_numericas])
atributos_teste_norm = atributos_teste.copy()
atributos_teste_norm[colunas_numericas] = modelo_normalizador.transform(
    atributos_teste[colunas_numericas])

# 8) HIPERPARAMETRIZAÇÃO: buscar os parâmetros da árvore de melhor acurácia
criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best', 'random']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=5)]
# profundidade limitada: árvores muito profundas "decoram" o treino
# (overfitting) e erram mais em pacientes novas
max_depth = [2, 3, 4, 5]
decision_tree_grid = {
    'criterion': criterion,
    'splitter': splitter,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
}
tree = DecisionTreeClassifier(random_state=42)
tree_hiperparametros = RandomizedSearchCV(
    estimator=tree,
    param_distributions=decision_tree_grid,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
)
tree_hiperparametros.fit(atributos_treino_norm, classes_treino_bal)
print('\nMelhores hiperparametros:', tree_hiperparametros.best_params_)
print('Acuracia media na validacao cruzada:', round(tree_hiperparametros.best_score_, 4))

# 9) TREINAR O MODELO DEFINITIVO COM OS MELHORES HIPERPARÂMETROS E SALVAR
tree = DecisionTreeClassifier(**tree_hiperparametros.best_params_, random_state=42)
breast_tree = tree.fit(atributos_treino_norm, classes_treino_bal)
with open('breast_tree.model', 'wb') as arquivo:
    dump(breast_tree, arquivo)
# a inferência precisa saber a ordem das colunas usadas no treinamento
with open('colunas_breast.model', 'wb') as arquivo:
    dump(list(dados_atributos.columns), arquivo)

# 10) AVALIAR A ACURÁCIA DO MODELO
acuracia_treino = accuracy_score(classes_treino_bal, breast_tree.predict(atributos_treino_norm))
acuracia_teste = accuracy_score(classes_teste, breast_tree.predict(atributos_teste_norm))
print('\nAcuracia nos dados de treino:', round(acuracia_treino, 4))
print('Acuracia nos dados de teste:', round(acuracia_teste, 4))
print('\nModelo salvo em breast_tree.model')
