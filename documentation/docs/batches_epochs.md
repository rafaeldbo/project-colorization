# **Batches e Epochs**

___
## ***Batches***

Os chamados **batches** são amostras agrupadas que são usadas no modelo a cada vez que os parâmetros dele mudam, ou seja, para cada conjunto de parâmetros, teremos um **batch** e assim por diante.

Ao final de cada **batch**, é calculado o erro utilizando uma *loss function* (como o MSE) e a partir desse erro, o algoritmo atualiza o modelo a fim de melhorá-lo.

O `batch size` é o hiperparâmetro que define o número de amostras por **batch** que serão utilizadas durante as iterações.

Existem três tipos de algoritmos de aprendizado com base no `batch size`, sendo eles:

1. **Gradiente Descendente por Batch**: quando o `batch size` é do tamanho da base de treinamento;

2. **Gradiente Descendente Estocástico**: quando o `batch size` é igual a 1;

3. **Gradiente Descendente por Mini-Batch**: quando o `batch size` é um valor entre 1 e o tamanho da base de treinamento (valor diferente de cada um dos extremos).

Em nosso modelo, por exemplo, utilizamos um `batch size` de 32 imagens, ou seja, utilizamos o algoritmo de **gradiente descendente por mini-batch** para aprendizagem do modelo.

Os **Batches** são criados a partir da separação de uma base de dados em conjuntos menores de dados por meio de um `DataLoader`.Normalmente, o `DataLoader` gera os **Batches** de forma aleatorizada e os carrega na memória (seja ela a RAM, cache, ou da própria GPU) para que o modelo possa utiliza-los. Esse processo de carregamento é potencialmente um dos mais lentos durante o treinamento, porém ele pode ser acelerado por meio da utlização do multiprocessamento que discutiremos mais para frente.

!!! example "Código: DataLoader"
    Um `DataLoader` pode ser declarado da seguinte forma:
    ``` python 
    from torch.utils.data import DataLoader, Dataset

    # Classe de um dataset personalizado capaz de carregar as imagens da base de dados
    class ImageDataset(Dataset):
        ...

    dataloader = DataLoader(
        ImageDataset(), # Dataset de imagens 
        batch_size=32, # tamanho dos batchs
        shuffle=True, # batchs formados aleatoriamente
        # ... parâmetros de multiprocessamento
    )
    ```
___
## ***Epochs***

Uma **epoch** (ou *época*) é o hiperparâmetro que define a quantidade de vezes na qual o algoritmo de aprendizagem anteriormente explicado será computado ao longo da amostra de treinamento. Para nos situarmos melhor, podemos imaginar que seria feito um `for-loop` para executar o treinamento por uma determinada quantidade de *epochs*.

!!! example "Código: For-loop com epochs"

    O código abaixo esboça um código de treinamento de um modelo utilizando multiplas Epochs e Batches:

    ```py
    # ... código acima
    # imports, Declaração do modelo, dataset, otimizador...

    for epoch in range(epochs):
        for i, data in enumerate(dataloader): 
            
            # Rotina de Treinamento

    # Finalização do Treino e começo do Teste
    # ... código abaixo
    ```