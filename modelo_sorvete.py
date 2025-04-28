import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulando alguns dados
temperatura = np.array([20, 22, 24, 26, 28, 30, 32, 34]).reshape(-1, 1)
vendas = np.array([200, 220, 250, 270, 300, 330, 360, 390])

# Criando o modelo
modelo = LinearRegression()
modelo.fit(temperatura, vendas)

# Fazendo uma previsão
temp_nova = np.array([[29]])
vendas_previstas = modelo.predict(temp_nova)
print(f'Previsão de vendas para 29°C: {vendas_previstas[0]:.0f} sorvetes')

# Visualizando
plt.scatter(temperatura, vendas, color='blue')
plt.plot(temperatura, modelo.predict(temperatura), color='red')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete')
plt.title('Previsão de Vendas de Sorvete x Temperatura')
plt.show()
