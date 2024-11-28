# **Otimizando os Parâmetros do Modelo**

Durante o treinamento do modelo, é necessário ajustar os parâmetros do modelo para minimizar o erro. Para isso, utilizamos técnicas de otimização que ajustam os parâmetros do modelo de acordo com o erro calculado por uma **fução de perda**.

___
## **Função de Perda**

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
___
## **Gradiente Decendente**

Após calcular o erro do modelo pela **função de perda**, é necessário ajustar os parâmetros do modelo para **minimizar** esse erro. A técnica de gradiente descendente irá determinar como devem ser feitos esses ajustes utilizando o **vetor gradiente** da função de perda. 

O vetor gradiente é calculado através das derivadas parciais da função e o seu valor em cada ponto indicará a direção em que o crescimento da função é máximo (ou seja, o inverso do vetor gradiente indica a direção em que está o mínimo da função).

Com a informação obtida através do vetor gradiente, pode-se determinar quais variáveis do modelo devem ser alteradas e se elas devem aumentar ou diminuir para resultar na minimização do erro.

O tamanho do passo que será dado para ajustar as variáveis do modelo é determinado pelo `learning rate`. Quanto maior o `learning rate`, maior será o passo e mais rápido o modelo irá convergir para o erro mínimo, porém, se o `learning rate` for muito grande, o modelo pode não convergir para o mínimo e oscilar em torno dele. Por outro lado, se o `learning rate` for muito pequeno, o modelo pode demorar muito para convergir para o mínimo.

Para entender melhor tudo isso, vamos supor que o resultado do modelo seja dado por uma função $f$ de três variáveis $a, b, c$ e que o modelo tem um `learning rate` de $α$.

$$f(a,b,c)$$

O vetor gradiente dessa função será dado por:

$$(\frac{∂f}{∂a}, \frac{∂f}{∂b}, \frac{∂f}{∂c})$$

Com o gradiente decendente a alteração das variáveis para chegar ao erro mínimo seria feita da seguinte forma:

$$a_{otim} = a - α\frac{∂f}{∂a}$$

$$b_{otim} = b - α\frac{∂f}{∂b}$$

$$c_{otim} = c - α\frac{∂f}{∂c}$$

Não parece complicado, mas no caso de redes neurais em que as variáveis de uma camada dependem das variáveis das camadas anteriores, o ajuste feito pelo **Gradiente Decendente** se torna mais complexo, pois requer que o cálculo das derivadas parciais da função seja propagado em todas as camadas da rede.Essa propagação é chamada de **back-propagation** e no nosso modelo será feita pela função **backward** do torch.

___
## **Back-Propagation**

Como explicado no tópico acima, em uma função de perda no formato $f(a,b,c)$, calcular a derivada da função de perda é suficiente para atualizar os parâmetros $a$, $b$ e $c$ de forma a minimizar o erro do modelo.

Entretanto, no caso da rede neural, o formato da função de perda é mais parecido com:

$$g(h(a,b), j(a,b), k(a,b))$$

Em que a função final depende de outras funções. Isso porque, cada camada, ao invés de receber diretamente parâmetros do modelo, recebem as saídas dos nós da camada anterior. Apenas a primeira camada recebe apenas parâmetros do modelo.

Dessa forma, para determinar o impacto de cada parâmetro no erro do modelo, apenas a derivada da função de perda não é suficiente, essa derivada deverá ser propagada por todas as camadas, isso é chamado **back-propagation**.

Vamos usar como exemplo a função, que seria o formato da função de perda de uma rede neural convolucional bem simples, com duas camadas.

$$g(h_1(a,b), h_2(c,d))$$

Em que as funções $h$ seriam as saídas dos nós da penúltima camada. Como nesse caso só temos 2 camadas, essa camada também é a que recebe os parâmetros do modelo.

Para obter o impacto dos parâmetros $a$, $b$, $c$ e $d$ no erro do modelo, precisamos da derivada da função de erro em relação a esses parâmetros, ou seja:

$$(\frac{∂g}{∂a}, \frac{∂g}{∂b}, \frac{∂g}{∂c}, \frac{∂g}{∂d})$$ 

Entretanto, essa informação não está disponível na última camada, uma vez que ela só receberá os valores das funções $h_1$ e $h_2$ e não o valor dos parâmetros. A derivada calculada na última camada seria:

$$(\frac{∂g}{∂h_1}, \frac{∂g}{∂h_2})$$

Entretanto, podemos continuar calculando a derivada para as camadas anteriores. A derivada na penúltima camada, em que as saídas são $h_1$ e $h_2$ e as entradas são $a$, $b$, $c$ e $d$, seria:

$$(\frac{∂h_1}{∂a}, \frac{∂h_1}{∂b}, \frac{∂h_2}{∂c}, \frac{∂h_2}{∂d})$$

Com esses resultados, conseguiríamos relacionar a variação da função do erro $g$ com os parâmetros do modelo $a$, $b$, $c$ e $d$, pois pela regra da cadeia temos que:

$$\frac{∂g}{∂a} = \frac{∂g}{∂h_1}\cdot\frac{∂h_1}{∂a}$$

$$\frac{∂g}{∂b} = \frac{∂g}{∂h_1}\cdot\frac{∂h_1}{∂b}$$

$$\frac{∂g}{∂c} = \frac{∂g}{∂h_2}\cdot\frac{∂h_2}{∂c}$$

$$\frac{∂g}{∂d} = \frac{∂g}{∂h_2}\cdot\frac{∂h_2}{∂d}$$

A propagação da derivada fará com que a derivada seja calculada em todas as camadas, começando da última e indo até a primeira camada, que é a que recebe os parâmetros do modelo diretamente. Só então, a informação obtida será suficiente para realizar o ajuste do modelo. Nesse caso, fizemos a derivada 2 vezes pois o exemplo era uma rede de 2 camadas.

___
## **Otimizador**

O otimizador é uma função que será utilizada para ajustar as variáveis do modelo durante a fase de treinamento, de forma a minimizar o erro do modelo. No nosso modelo estamos utilizando o otimizador **Adam**, que também utiliza o vetor gradiente da função de perda, mas é mais complexo do que a simples aplicação do **gradiente descentente** e permitirá um ajuste mais rápido.

No ajuste feito pelo Adam, além de também ser utilizada a segunda derivada da função de perda, a influência do vetor gradiente é ponderada por um fator que decai exponencialmente a cada iteração, tornando as variáveis menos voláteis.

O ajuste feito pelo Adam, para um modelo com learning rate α cujo resultado é uma função $f(a,b,c)$ seria como mostrado abaixo:

Teríamos a variável $a$ ajustada da seguinte forma:

$$a_{otim} = a - \frac{αV}{\sqrt{S}+ε}$$

Em que:

$$V = \frac{β1S + (1-β1)\frac{∂f}{∂a}}{1-{β1}^{t}}$$

$$S = \frac{β2S + (1-β2)\frac{∂f}{{∂a}^{2}}}{1-{β2}^{t}}$$

Os valores $β1$, $β2$ e $ε$ são constantes, geralmente definida com os seguintes valores:

$β1 = 0.9$

$β2 = 0.999$

$ε = {10}^{-8}$

E $t$ é o número de iterações já feitas no modelo.

!!! example "Código: Otimização do modelo

___
## **Referências**

1. [What is Adam Optimizer?](https://www.geeksforgeeks.org/adam-optimizer/)
2. [Optimization techniques for Gradient Descent](https://www.geeksforgeeks.org/optimization-techniques-for-gradient-descent/)
3. [Exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing)
5. [PyTorch: Connection Between loss.backward() and optimizer.step()](https://www.geeksforgeeks.org/pytorch-connection-between-lossbackward-and-optimizerstep/)
4. [Gradient Descent in Linear Regression](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/)