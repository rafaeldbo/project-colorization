## ***Batches***

Os chamados *batches* são amostras agrupadas que são usadas no modelo a cada vez que os parâmetros dele mudam, ou seja, para cada conjunto de parâmetros, teremos um *batch* e assim por diante.

Ao final de cada *batch*, é calculado o erro utilizando uma *loss function* (como o MSE) e a partir desse erro, o algoritmo atualiza o modelo a fim de melhorá-lo.

O `batch size` é o hiperparâmetro que define o número de amostras por *batch* que serão utilizadas durante as iterações.

Existem três tipos de algoritmos de aprendizado com base no `batch size`, sendo eles:

$(i)$ gradiente descendente por *batch*: quando o `batch size` é do tamanho da base de treinamento;

$(ii)$ gradiente descendente estocástico: quando o `batch size` é igual a 1 e

$(iii)$ gradiente descendente por *mini-batch*: quando o `batch size` é um valor entre 1 e o tamanho da base de treinamento (valor diferente de cada um dos extremos).

Em nosso modelo, por exemplo, utilizamos um `batch size` de 32 imagens, ou seja, utilizamos o algoritmo $(iii)$ para aprendizagem do modelo.

## ***Epochs***

Uma *epoch* (ou época) é o hiperparâmetro que define a quantidade de vezes na qual o algoritmo de aprendizagem anteriormente explicado será computado ao longo da amostra de treinamento. Para nos situarmos melhor, podemos imaginar que seria feito um `for-loop` para executar o treinamento por uma determinada quantidade de *epochs*.

??? example "For-loop com epochs"

    Como exemplo, o código abaixo representa uma forma de implementar o *loop* com *epochs*:

    ```py
    from torch import nn

    # ... código acima
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for i, data in enumerate(dataloader): # dataloader -> objeto que carrega os dados da base de treinamento
            # ---
            # configurações adicionais; serão abordadas posteriormente
            gray, color, category = data

            gray = gray.to(device)
            color = color.to(device) # target
            category = category.to(device)

            optimizer.zero_grad()
            # ---

            outputs = ecnn(gray, category) # input tensor

            loss = criterion(outputs, color) # função de perda MSE
            loss.backward() # back-propagation
            optimizer.step() # forward

            print(f"[{epoch+1:2d}, {i + 1:3d}] loss: {loss.item()/2000:.3f}")

    # ... código abaixo
    ```