import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

tabela = pd.read_csv("advertising.csv")
display(tabela)

#criar o grafico
sns.pairplot(tabela)
# exibir o grafico
plt.show()

#criar o grafico
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)
# exibir o grafico
plt.show()
# separar os dados em x e y
y = tabela["Vendas"]
x = tabela[["TV","Radio","Jornal"]]

# separar os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1)


# cria a inteligencia
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina a inteligencia
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# cria as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# compara as previsoes com o gabarito
print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))
print(f"{r2_score(y_teste, previsao_regressaolinear):.1%}")
print(f"{r2_score(y_teste, previsao_arvoredecisao):.1%}")
# O melhor modelo é o da Arvore de decisão
# Novas previsoes
novos_valores = pd.read_csv("novos.csv")
display(novos_valores)
nova_previsao = modelo_arvoredecisao.predict(novos_valores)
display(nova_previsao)