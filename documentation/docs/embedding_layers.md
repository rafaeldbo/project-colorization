# **Embeddings Layers**

`Embeddings` são representações lineares e condensadas de dados discretos (como palavras ou categorias), esses dados são traduzidos em uma fórmula matemática, de forma a enfatizar ou não características que eles podem possuir, tornando possível, assim, agrupar dados com caracteriscas similares. 

Utilizaremos `Embeddings` para treinar nosso modelo a reconhecer caracteristicas comuns em certos tipos de imagens, utilizando as categorias que citamos anteriormente. Isso nos será util, pois, uma vez que o modelo aprenda que que imagens categorizadas com sendo de **ambientes externos** possuem a caracteristica de ter um céu, e que esse céu costuma ser azul, ele colorirá o céu mais facilmente.

Para criar uma camada de `Embeddings` precisamos especificar quantos tipos de dados diferentes possuimos, quantas categorias possuimos por exemplo, e quantas caracteristicas cada `Embedding` terá.

``` python title="Criando uma camada de Embeddings"
from torch import nn

# criando uma camada de Embeddings com 8 embeddins e 10 carcateristicas
embd = nn.Embedding(8, 10)
```