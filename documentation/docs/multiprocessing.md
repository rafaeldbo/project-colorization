O multiprocessamento é utilizado para que o nosso programa seja executado de maneira mais eficiente, visto que a quantidade de dados de entrada é grande e um fator importante para tornar a execução do algoritmo viável é a velocidade com a qual ele será executado.

Dessa forma, a biblioteca do PyTorch disponibiliza ferramentas de paralelização quando carregamos nossos dados. O objeto `DataLoader` possui atributos que permitem o uso de multiprocessamento, os principais são:

1. `num_workers`: define a quantidade de subprocessos iremos utilizar para carregar os dados;
2. `pin_memory`: define se os dados serão copiados para o espaço da memória fixado do dispositivo (CPU ou GPU) para transferência dos dados do modelo; 
3. `prefetch_factor`: define a quantidade de **batches** que serão carregados antecipadamente por cada subprocesso.

Os parâmetros citados acima são recomendados principalmente quando estamos treinando o modelo utilizando a GPU e com uma quantidade razoável de **batches** por **epoch**. Nessas condições, a combinação certa desses parâmetros tornará a execução do código mais eficiente, economizando tempo de processamento.

Os subprocessos, após várias iterações, consumirão a mesma quantidade de memória da CPU que o processo pai. Isso justifica possíveis momentos em que a execução parece "engasgar" quando tratamos de um conjunto de treinamento grande e/ou quando são utilizados grandes números de subprocessos. Existem soluções para esse tipo de problema que são abordados da documentação da biblioteca (vide referência).

___
## **Referências**

1. [PyTorch documentation > torch.utils.data](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)