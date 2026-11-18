# Classificador Breast Cancer

Classifica se uma paciente tem indicativo de recorrência do câncer de mama, que é o atributo `Class`. A base usada é a [Breast Cancer da UCI](https://archive.ics.uci.edu/dataset/14/breast+cancer), com 286 pacientes e 9 atributos.

## Como rodar

Execute os comandos dentro desta pasta:

```bash
pip install -r requirements.txt
python treinamento.py
python inferencia.py
```

`treinamento.py` treina e salva o modelo. `inferencia.py` classifica uma nova paciente.

## Módulo de treinamento (`treinamento.py`)

1. **Carregar os dados** do arquivo `breast-cancer.csv`.
2. **Tratar os valores faltantes:** os `?` em node-caps (8) e breast-quad (1) são trocados pela moda da coluna.
3. **Separar** os atributos da classe.
4. **Transformar os atributos categóricos em números:**
   - Faixas com ordem (age, tumor-size, inv-nodes): cada faixa vira o seu valor do meio. Por exemplo, `50-59` vira 54,5.
   - Nominais (menopause, node-caps, breast, breast-quad, irradiat): viram colunas dummy com 0 ou 1.
5. **Separar treino (70%) e teste (30%)**, mantendo a proporção das classes.
6. **Balancear** o treino com SMOTE. A base tem 201 casos `no-recurrence-events` e 85 `recurrence-events`.
7. **Normalizar** os atributos numéricos com MinMaxScaler. As colunas dummy já estão entre 0 e 1.
8. **Hiperparametrizar** a DecisionTree com RandomizedSearchCV (50 combinações, validação cruzada com 5 partes), para obter o modelo de melhor acurácia.
9. **Treinar o modelo definitivo** com os melhores hiperparâmetros e salvar:
   - `breast_tree.model`: o modelo.
   - `normalizador_breast.model`: o normalizador.
   - `colunas_breast.model`: as colunas usadas no treino.
10. **Avaliar a acurácia** nos dados de treino e nos dados de teste.

### Resultado

| Item | Valor |
|------|-------|
| Melhores hiperparâmetros | criterion = entropy, splitter = best, max_depth = 5, min_samples_split = 10 |
| Acurácia na validação cruzada | 73,1% |
| Acurácia no treino | 77,3% |
| Acurácia no teste | 68,6% |

## Módulo de inferência (`inferencia.py`)

1. Recebe pelo teclado os 9 atributos da nova paciente.
2. Aplica as mesmas transformações do treinamento: faixas viram números, nominais viram dummies e os numéricos são normalizados.
3. Mostra o indicativo de diagnóstico (`recurrence-events` ou `no-recurrence-events`).
