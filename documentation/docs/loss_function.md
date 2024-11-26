# **Função de Perda**

As **funções de perda**, ou *loss functions*, são métricas utilizadas em Machine Learning para medir a __performance__ de um modelo utilizando os valores que foram previstos por ele em comparação aos valores reais da base de dados.

Uma função muito comumente usada em diversos modelos de previsão (e será a que utilizaremos nesse tutorial) é o **Mean Squared Error** (`MSE`), que é calculado pela diferença da valor obtido pela previsão e o valor real presente na base de dados. A equação dessa métrica é dada por:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat y_i)^2$$

Sendo $y_i$ o valor real e $\hat y_i$ o valor previsto.

!!! example "Exemplo: Função de Perda"
    Um exemplo simples da utilização da função de perda pode ser visto abaixo:

    ```py
    from torch import nn
    
    criterion = nn.MSELoss() # Declaração da função de perda

    y = torch.randn(3, 5) # tensor aleatorio
    y_pred = torch.randn(3, 5, requires_grad=True) # tensor aleatorio

    loss = criterion(y_pred, y) # calculo da perda com MSE
    loss.backward() # Obtenção da derivada
    ```