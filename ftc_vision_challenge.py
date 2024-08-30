# Importação das bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregamento e Preparação do Conjunto de Dados

# Carregar o conjunto de dados CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar as imagens (valores de 0 a 255 para 0 a 1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definição dos nomes das classes
class_names = ['Avião', 'Carro', 'Pássaro', 'Gato', 'Cervo',
               'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

# Visualização de algumas imagens de exemplo
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # Rótulo da classe
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# 2. Construção do Modelo CNN

# Inicialização do modelo sequencial
model = models.Sequential()

# Primeira camada convolucional com 32 filtros, tamanho de kernel 3x3 e ativação ReLU
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Primeira camada de pooling
model.add(layers.MaxPooling2D((2, 2)))

# Segunda camada convolucional com 64 filtros
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Segunda camada de pooling
model.add(layers.MaxPooling2D((2, 2)))

# Terceira camada convolucional com 64 filtros
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Camada de flatten para transformar a saída 3D em 1D
model.add(layers.Flatten())
# Camada densa totalmente conectada com 64 neurônios
model.add(layers.Dense(64, activation='relu'))
# Camada de saída com 10 neurônios (uma para cada classe)
model.add(layers.Dense(10))

# Resumo da arquitetura do modelo
model.summary()

# 3. Compilação e Treinamento do Modelo

# Compilação do modelo com otimizador Adam e função de perda de entropia cruzada
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treinamento do modelo por 10 épocas com validação no conjunto de testes
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# 4. Avaliação do Modelo

# Avaliação do desempenho do modelo no conjunto de testes
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nAcurácia no conjunto de testes: {test_acc:.2f}')

# 5. Visualização das Previsões

# Adiciona a camada Softmax para converter logits em probabilidades
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
# Realiza previsões no conjunto de testes
predictions = probability_model.predict(test_images)

# Função para plotar a imagem com a previsão
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):.2f}% ({class_names[true_label[0]]})", color=color)

# Função para plotar o array de valores das previsões
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Exemplo de visualização para as primeiras 5 imagens do conjunto de testes
num_rows = 5
num_cols = 2
num_images = num_rows

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    # Plot da imagem com a previsão
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    # Plot do array de valores das previsões
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
