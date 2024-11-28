# Testando o Modelo

Após o treinamento do modelo, devemos testar nosso modelo em um conjunto de imagem completamente novo para sabermos se ele é capaz de generalizar bem a colorização para imagens que ele nunca viu antes. Para isso, utilizaremos o conjunto de teste que foi separado anteriormente.

Nosso modelo treinado com `50 Epochs`, `5000 imagens` e um **learning rate** de `0.01` foi capaz de atingir um `MSE` de $142.98$ no conjunto de treino e $174,64$ no conjunto de teste 

![Resultados do teste](./img/output_ecnn_advanced_5000_0.png)