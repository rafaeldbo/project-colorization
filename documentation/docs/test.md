Após o treinamento do modelo, ele será testado utilizando 739 imagens diversas que foram classificadas nas categorias existentes (comida, pessoas, etc.).

Para isso, utilzaremos um `batch_size` de 4 amostras quando carregamos os nossos dados.
Carregamos o modelo treinado para que ele receba as imagens da base de teste e colorize-as de acordo.

Apesar da grande quantidade de imagens, não é necessária nenhuma paralelização, visto que o modelo já está treinado e será executado sobre os dados que disponibilizamos para ele.

Após finalizar a execução do teste, os resultados do teste são mostrados pela imagem a seguir, que é salva em um diretório (`outputs`):

![Resultados do teste](./img/output_ecnn_advanced_5000_0.png)