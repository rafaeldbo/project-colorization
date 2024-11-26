O multiprocessamento é utilizado para que o nosso programa seja executado de maneira mais eficiente, visto que a quantidade de dados de entrada é grande e um fator importante para tornar a execução do algoritmo viável é a velocidade com a qual ele será executado.

Dessa forma, a biblioteca do PyTorch disponibiliza ferramentas de paralelização quando carregamos nossos dados. O objeto `DataLoader` possui atributos que dizem respeito à quantidade de subprocessos iremos utilizar (`num_workers`) ou que definem se os Tensors serão copiados para a memória fixada (`pin_memory`), quantidade de *batches* que serão carregadas em cada subprocesso (`prefetch_factor`) entre outros.

Os três citados anteriormente serão utilizados em nossa modelagem para que o código seja executado de maneira mais eficiente.

Os subprocessos, após várias iterações, consumirão a mesma quantidade de memória da CPU que o processo pai. Isso justifica possíveis momentos em que a execução parece "engasgar" quando tratamos de um conjunto de treinamento grande e/ou quando são utilizados grandes números de subprocessos. Existem soluções para esse tipo de problema que são abordados da documentação da biblioteca (ref).

## **Referências**

1. [PyTorch documentation > torch.utils.data](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)