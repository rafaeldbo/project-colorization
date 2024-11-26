## **Função de perda**

As funções de perda, ou *loss functions*, são métricas utilizadas em Machine Learning para medir a performance de um modelo utilizando os valores que foram previstos por ele em comparação aos valores reais de nosso *dataset*.

Uma função muito comumente usada em diversos modelos de previsão é o *Mean Squared Error* (MSE) é calculado pela diferença da valor obtido pela previsão e o valor real na tabela de dados. A equação dessa métrica é dada por:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat y_i)^2
$$

Sendo $y_i$ o valor real e $\hat y_i$ o valor previsto.

??? example "Exemplo de utilização"
    Em Python, utilizando a biblioteca pytorch, teríamos:

    ```py
    from torch import nn
    
    loss = nn.MSELoss()
    input_ = torch.randn(3, 5, requires_grad=True) # -> Tensor
    target = torch.randn(3, 5) # -> Tensor
    output = loss(input_, target) 
    output.backward()
    ```