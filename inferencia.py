# MÓDULO DE INFERÊNCIA - recebe uma nova paciente e indica o diagnóstico (Class)
import pandas as pd
from pickle import load

# 1) Carregar o normalizador, o modelo e a lista de colunas salvos no treinamento
with open('normalizador_breast.model', 'rb') as arquivo:
    normalizador = load(arquivo)
with open('breast_tree.model', 'rb') as arquivo:
    breast_tree = load(arquivo)
with open('colunas_breast.model', 'rb') as arquivo:
    colunas = load(arquivo)

colunas_faixas = ['age', 'tumor-size', 'inv-nodes']
colunas_nominais = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']
colunas_numericas = colunas_faixas + ['deg-malig']

# valores possíveis de cada atributo, mostrados para quem for digitar
opcoes = {
    'age': 'idade em faixa: 20-29, 30-39, 40-49, 50-59, 60-69, 70-79',
    'menopause': 'menopausa: lt40, ge40, premeno',
    'tumor-size': 'tamanho do tumor em faixa: 0-4, 5-9, 10-14, ..., 50-54',
    'inv-nodes': 'linfonodos afetados em faixa: 0-2, 3-5, 6-8, ..., 24-26',
    'node-caps': 'node-caps: yes, no',
    'deg-malig': 'grau de malignidade: 1, 2, 3',
    'breast': 'mama: left, right',
    'breast-quad': 'quadrante: left_up, left_low, right_up, right_low, central',
    'irradiat': 'fez radioterapia: yes, no',
}


def faixa_para_numero(faixa):
    inicio, fim = faixa.split('-')
    return (int(inicio) + int(fim)) / 2


# 2) Receber os dados da nova paciente
print('Digite os dados da nova paciente:')
nova_paciente = {}
for atributo in opcoes:
    nova_paciente[atributo] = input(f'{atributo} ({opcoes[atributo]}): ').strip()
nova_paciente = pd.DataFrame([nova_paciente])

# 3) Aplicar as mesmas transformações do treinamento
# faixas -> valor do meio
for coluna in colunas_faixas:
    nova_paciente[coluna] = nova_paciente[coluna].apply(faixa_para_numero)
nova_paciente['deg-malig'] = nova_paciente['deg-malig'].astype(int)
# nominais -> colunas dummy, na mesma ordem do treinamento
# (as dummies das categorias que a paciente não tem ficam com 0)
nova_paciente = pd.get_dummies(nova_paciente, columns=colunas_nominais, dtype=int)
nova_paciente = nova_paciente.reindex(columns=colunas, fill_value=0)
# normalizar os atributos numéricos com o normalizador salvo
nova_paciente[colunas_numericas] = normalizador.transform(nova_paciente[colunas_numericas])

# 4) Classificar a nova paciente
diagnostico = breast_tree.predict(nova_paciente)[0]
print('\nIndicativo de diagnostico (Class):', diagnostico)
if diagnostico == 'recurrence-events':
    print('O modelo indica RECORRENCIA do cancer de mama.')
else:
    print('O modelo indica que NAO HA RECORRENCIA do cancer de mama.')
