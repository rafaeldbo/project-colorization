# **Normalização em Batches e Funções de Ativação** 

___
## **Normalização em Batches** (Batch normalization)

A **normalização em Batches** (*Lotes*) surgiu com o propósito de remediar o **problema de mudança de covariável interna** (em inglês, *internal covariant shift problem*). Esse problema é causado pela variação da distribuição dos parâmetros das entradas, que pode dificultar na convergência do modelo, uma vez que torna o processo de treinamento mais lento do que deveria ser.

Quando é feita a **normalização em mini-batches** (*mini-lotes*), isto é, quando a média é ajustada para 0 e a variância para 1, são utilizados dois novos parâmetros, o deslocamento e a escala, que servem para otimizar a normalização para as ativações. Dessa forma, o processo de aprendizado da rede neural é estabilizado, reduzindo a **mudança de covariável interna** e garantindo consistência entre as camadas do modelo.

Para criar uma camada de **normalização em Batches** é obrigatório apenas especificar a quantidade de características (*features*) que serão normalizadas. No caso das imagens, essas características podem ser os layers da imagem.

```py
from torch import nn

bn = nn.BatchNorm2d(32)
```

___
## **Funções de Ativação** (Activation Functions)

Em cada camada, seja ela de convolução ou convolução transposta, teremos vários nós. Como é característico da rede neural, um nó receberá como entrada os valores de saída de nós da camada anterior. O valor de saída do nó atual dependerá de uma soma de um valor "bias" com os valores ponderados da camada anterior. 

Entretanto, esse valor de saída será ajustado por uma **função de ativação**, que determina se o valor calculado irá ou não ser propagado para a próxima camada. No nosso modelo, a função de ativação é a `ReLU` (Rectified Linear Unit), que determina que a saída do nó será a soma calculada caso ela seja maior do que 0, e 0 caso contrário. Essa função de ativação reduz o **problema do desaparecimento de gradientes**, que causaria perda de informação na rede.

Uma vez que o aprendizado do modelo é justamente determinar os pesos ideais para essa soma, o valor de saída do nó deverá mudar a cada interação.

___
## **Referências**

1. [What is Batch Normalization In Deep Learning?](https://www.geeksforgeeks.org/what-is-batch-normalization-in-deep-learning/)
2. [Internal Covariant Shift Problem in Deep Learning](https://www.geeksforgeeks.org/3.internal-covariant-shift-problem-in-deep-learning/)
3. [Activation functions in Neural Networks](https://www.geeksforgeeks.org/activation-functions-neural-networks/)
4. [Understanding the Rectified Linear Unit (ReLU): A Key Activation Function in Neural Networks](https://medium.com/@meetkp/understanding-the-rectified-linear-unit-relu-a-key-activation-function-in-neural-networks-28108fba8f07)