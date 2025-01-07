# Examen de IA

## Parte 1: Evaluación de Modelos de IA (Teoría)

### Pregunta 1:
Explica qué son los diferentes tipos de capas (layers) en redes neuronales. ¿Para qué se utiliza cada una de ellas?

**Respuesta:**
Las capas de una red neuronal son los bloques básicos que definen la arquitectura de la red. Los tipos más comunes son:

1. **Capa lineal (Dense o Fully Connected Layer):**
   - Es una capa en la que cada neurona está conectada a todas las neuronas de la capa anterior.
   - Se utiliza para transformaciones lineales y generalmente en las etapas finales de la red para la clasificación o regresión.

   ![https://pysource.com/wp-content/uploads/2022/08/flatten-and-dense-layers-computer-vision-with-keras-p-6-dense-layer-scheme-1024x723.jpg](https://pysource.com/wp-content/uploads/2022/08/flatten-and-dense-layers-computer-vision-with-keras-p-6-dense-layer-scheme-1024x723.jpg)

2. **Capa convolucional (Convolutional Layer - Conv):**
   - Se utiliza en redes neuronales convolucionales (CNN). Es ideal para procesamiento de imágenes.
   - En esta capa, los filtros (kernels) recorren la imagen (o las características previas) para extraer patrones locales.
      - **El kernel** (también llamado filtro) es una matriz pequeña de números (pesos) que se desplaza sobre la entrada (imagen, por ejemplo) para extraer características específicas como bordes, texturas o patrones. **Extrae patrones locales específicos.**
      - **Max Pooling**: Es una técnica de reducción de dimensionalidad que toma una región específica del mapa de características (por ejemplo, 2x2) y selecciona el valor máximo dentro de esa región. **Reduce la complejidad y el tamaño del mapa de características, conservando las características más importantes.**

   ![https://miro.medium.com/v2/resize\:fit:1400/1\*uAeANQIOQPqWZnnuH-VEyw.jpeg](https://miro.medium.com/v2/resize\:fit:1400/1*uAeANQIOQPqWZnnuH-VEyw.jpeg)

3. **Capa de activación ReLU (Rectified Linear Unit):**
   - Función de activación que convierte valores negativos en cero y deja pasar valores positivos tal como están.
   - Se utiliza en redes profundas porque ayuda a evitar el problema de gradientes desaparecidos.

   ![https://jacar.es/wp-content/uploads/2023/03/FuncionRELU.png](https://jacar.es/wp-content/uploads/2023/03/FuncionRELU.png)

4. **Capa de activación Sigmoid:**
   - Función de activación que mapea los valores entre 0 y 1. Es útil en problemas de clasificación binaria. El dibujo debe incluir su curva suave en forma de "S"

   ![https://jacar.es/wp-content/uploads/2023/03/FuncionSigmoide.png](https://jacar.es/wp-content/uploads/2023/03/FuncionSigmoide.png)

5. **Capa de salida:**
   - Dependiendo del tipo de problema, puede ser una capa lineal para regresión o una capa con activación Softmax para clasificación múltiple.

### Pregunta 2:
Dibuja una red neuronal simple (con al menos 3 capas) e indica cómo se aplica el **Gradient Descent** en el entrenamiento.

**Respuesta:**
(Si es un examen en papel, el dibujo podría ser algo así):

[Entrada -> Capa Oculta (ReLU) -> Capa Oculta (ReLU) -> Capa de Salida (Sigmoid)]

- **Gradient Descent:** Es un método de optimización que ajusta los parámetros de un modelo (pesos) siguiendo la dirección del gradiente negativo de la función de costo. Esto minimiza la función de costo iterativamente.

   ![https://miro.medium.com/max/1400/1\*tQTcGTLZqnI5rp3JYO\_4NA.png](https://miro.medium.com/max/1400/1*tQTcGTLZqnI5rp3JYO_4NA.png)

- **Función de costo:** Mide cuán lejos está el modelo de la salida deseada. Ejemplo común: MSE (error cuadrático medio).
   - El dibujo debe mostrar una curva de la función de costo y una bola descendiendo por la pendiente hacia el mínimo global.

   ![https://miro.medium.com/v2/resize:fit:810/1*UUHvSixG7rX2EfNFTtqBDA.gif](https://miro.medium.com/v2/resize:fit:810/1*UUHvSixG7rX2EfNFTtqBDA.gif)


## Parte 2: Aprendizaje (Teoría)

### Pregunta 3:
¿Qué es el **Gradient Descent** y cómo se calcula el gradiente?

**Respuesta:**

1. **Gradiente Descendente:** Explica con tus palabras el flujo del gradiente descendente en un modelo de red neuronal. Incluye los pasos de:

   - Forward pass.
   - Backward pass (backpropagation).
   - Actualización de pesos.

   **Respuesta:**

   - En el **forward pass**, los datos de entrada se procesan a través de la red capa por capa, calculando las salidas en cada etapa.
   - En el **backward pass**, calculamos el gradiente de la función de costo respecto a los pesos utilizando la regla de la cadena.
   - Los pesos se actualizan con una regla como: , donde  es la tasa de aprendizaje y  el gradiente de los pesos respecto al costo.

   El **Gradient Descent** es un algoritmo de optimización utilizado en aprendizaje automático y redes neuronales para minimizar una función de coste. El algoritmo calcula el gradiente de la función de coste, que es la derivada de la función con respecto a los parámetros del modelo (pesos y sesgos en una red neuronal). 

   El gradiente es un vector que indica la dirección de mayor pendiente de la función. Al movernos en la dirección opuesta al gradiente (por eso "descenso"), buscamos el valor mínimo de la función de coste.

   El proceso de cálculo es el siguiente:
   1. Inicializamos los pesos de la red con valores aleatorios.
   2. Calculamos la salida del modelo y la diferencia con la salida esperada (error).
   3. Calculamos el gradiente de la función de coste respecto a cada peso.
   4. Actualizamos los pesos usando la fórmula:  
      $[ w = w - \eta \cdot \nabla_w J(w) ]$

      Donde:
      - $( \eta )$ es la tasa de aprendizaje.
      - $( \nabla_w J(w) )$ es el gradiente de la función de coste $( J(w) )$.

### Pregunta 4:
¿Qué es una **función de coste** y cómo se usa en el entrenamiento de modelos?

**Respuesta:**
La **función de coste** (o función de pérdida) es una medida de la discrepancia entre las predicciones de un modelo y los valores reales. El objetivo del entrenamiento es minimizar esta función, lo que significa reducir el error entre la predicción y la realidad.

- En problemas de clasificación binaria, se usa comúnmente la **entropía cruzada**.
- En regresión, se usa el **error cuadrático medio (MSE)**.

Durante el entrenamiento, la función de coste guía el ajuste de los pesos utilizando el **Gradient Descent**, con el objetivo de encontrar los parámetros que minimicen el coste.


## Parte 3: Programación

### Pregunta 5:
**Primera parte:**
Explica cómo funciona el **Gradient Descent** en el entrenamiento de un modelo, y escribe el **train loop** en pseudocódigo.

**Respuesta:**
En el **Gradient Descent** se ajustan los pesos de los modelos iterativamente en función del gradiente de la función de coste. El pseudocódigo básico para un entrenamiento con **Gradient Descent** sería:

```python
# Parámetros iniciales
learning_rate = 0.01
epochs = 1000
weights = inicializar_pesos()
bias = inicializar_bias()

# Loop de entrenamiento
for epoch in range(epochs):
    for X_batch, y_batch in batch_de_datos:
        # Paso hacia adelante
        predicciones = modelo(X_batch, weights, bias)
        
        # Calcular la función de coste
        coste = funcion_de_coste(predicciones, y_batch)
        
        # Calcular el gradiente
        gradiente_pesos, gradiente_bias = calcular_gradientes(X_batch, y_batch, predicciones)
        
        # Actualizar los pesos y el sesgo
        weights -= learning_rate * gradiente_pesos
        bias -= learning_rate * gradiente_bias
    
    # Mostrar el coste por cada época
    print(f"Epoch {epoch+1}, Coste: {coste}")
```

Este train loop sigue el proceso de:

   1. Propagación hacia adelante: calcular las predicciones con los pesos actuales.
   2. Cálculo de la función de coste: evaluar la diferencia entre la predicción y la realidad.
   3. Propagación hacia atrás: calcular el gradiente de la función de coste.
   4. Actualización de los pesos y sesgos.

**Segunda parte:**

Dado un archivo CSV con datos de ventas: `ventas.csv` con columnas `fecha`, `producto`, y `cantidad`, escribe un código en Python que:

- Lea el archivo.
- Calcule el total de ventas por producto.

**Respuesta:**

```python
import pandas as pd

# Leer el archivo CSV
datos = pd.read_csv('ventas.csv')

# Calcular el total de ventas por producto
ventas_por_producto = datos.groupby('producto')['cantidad'].sum()

print(ventas_por_producto)
```

**Train Loop:** Escribe un train loop para entrenar un modelo simple usando PyTorch.

**Respuesta:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Modelo simple
model = nn.Linear(1, 1)  # Una capa lineal
loss_function = nn.MSELoss()  # Error cuadrático medio
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Gradiente descendente estocástico

# Datos de entrenamiento (ejemplo simple)
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Train loop
for epoch in range(100):
      model.train()
      optimizer.zero_grad()  # Reiniciar gradientes
      predictions = model(x_train)  # Forward pass
      loss = loss_function(predictions, y_train)  # Calcular la pérdida
      loss.backward()  # Backward pass
      optimizer.step()  # Actualizar los pesos
      if epoch % 10 == 0:
         print(f'Epoch {epoch}, Loss: {loss.item()}')
```

---

## Parte 4: Matemáticas (Derivada y Gradiente)

### Pregunta 6:
Calcula la derivada de la función de activación Sigmoid.

**Respuesta:**
La función de activación Sigmoid es:
$[\sigma(x) = \frac{1}{1 + e^{-x}}]$

La derivada de la Sigmoid es:
$[\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))]$

Esto se puede comprobar a través de la regla de la cadena. La derivada de la función Sigmoid se utiliza durante el cálculo del gradiente en el algoritmo de **backpropagation**.

---

### Pregunta 7:
¿Cuál es la diferencia entre el **gradiente** y la **derivada** en el contexto del aprendizaje profundo?

**Respuesta:**
En el contexto de redes neuronales:
- **Derivada:** La derivada de una función mide la tasa de cambio de la función respecto a una variable. En el caso de redes neuronales, la derivada se usa para calcular cómo cambia el valor de la función de coste con respecto a los pesos.
- **Gradiente:** El gradiente es un vector que contiene todas las derivadas parciales de la función de coste con respecto a cada peso del modelo. En **Gradient Descent**, el gradiente se utiliza para actualizar todos los parámetros (pesos y sesgos) simultáneamente.

En resumen, la derivada es un caso unidimensional, mientras que el gradiente es un caso multidimensional (cuando se trata de varios pesos en una red neuronal).
