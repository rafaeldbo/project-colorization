# **Tipo de Modelo**

O modelo desenvolvido é do tipo `Autoencoder`, ou seja, ele terá duas etapas chamadas **Encoder** e **Decoder**. No **Encoder**, a informação de entrada é __reduzida__ para uma dimensão menor enquanto no **Decoder** essa informação reduzida será utilizada para __reconstruir__ a informação de saída com o __tamanho original__. Há diferentes formas de realizar o desenvolvimento necessário nessas duas etapas, nesse caso cada etapa será desenvolvida usando uma `Rede Neural Convolucional`.

A `Rede Neural Convolucional` é um tipo de Rede Neural, dessa forma, a informação irá fluir da entrada para a saída passando por camadas de nós, em que cada nó transforma a informação recebida e passa adiante. 

A diferença entre a `rede neural convolucional` e a regular está em como essa informação é passada. Na rede neural regular, todos os nós de uma camada estão ligados com todos os nós da camada seguinte, já na convolucional, as informações de um nó são passados para apenas alguns dos nós da camada seguinte, permitindo que cada parte da informação seja analisada separadamente. 

No caso da imagem, a separação da informação é feita pelo **field** que será explicado mais adiante. Essa separação é muito utilizada na análise de imagens pois, com isso, apenas a informação necessária para identificar um determinado padrão é passada adiante, reduzindo o número de parâmetros do modelo e tornando-o mais eficiente.

___
## **Referências**
1. [Convolutional Neural Networks for Computer Vision](https://medium.com/thedeephub/convolutional-neural-networks-for-computer-vision-a913e77c60ff)

2. [Generative Models and Autoencoders](https://medium.com/@geokam/building-an-image-colorization-neural-network-part-1-generative-models-and-autoencoders-d68f5769d484)