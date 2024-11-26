# **Otimizador**

O otimizador é uma função que será utilizada para ajustar as variáveis do modelo durante a fase de treinamento, de forma a minimizar o erro do modelo. O **gradiente decendente** é um otimizador. No nosso modelo estamos utilizando o otimizador **Adam**, que também utiliza o vetor gradiente da função de perda, mas é mais complexo e permitirá um ajuste mais rápido.

No ajuste feito pelo Adam, além de também ser utilizada a segunda derivada da função de perda, a influência do vetor gradiente é ponderada por um fator que decai exponencialmente a cada iteração, tornando as variáveis menos voláteis.

O ajuste feito pelo Adam, para um modelo com learning rate α cujo resultado é uma função $f(a,b,c)$ seria como mostrado abaixo:

Teríamos a variável $a$ ajustada da seguinte forma:

$$a = a - \frac{αV}{\sqrt{S}+ε}$$

Em que:

$$V = \frac{β1S + (1-β1)\frac{∂f}{∂a}}{1-{β1}^{t}}$$

$$S = \frac{β2S + (1-β2)\frac{∂f}{{∂a}^{2}}}{1-{β2}^{t}}$$

Os valores $β1$, $β2$ e $ε$ são constantes, geralmente definida com os seguintes valores:

$β1 = 0.9$

$β2 = 0.999$

$ε = {10}^{-8}$

E $t$ é o número de iterações já feitas no modelo.
___
## **Gradiente Decendente**

Na fase de treinamento, após calcular o erro do modelo pela **função de perda**, é necessário ajustar os parâmetros do modelo para **minimizar** esse erro. A técnica de gradiente descendente irá determinar como devem ser feitos esses ajustes utilizando o **vetor gradiente** da função de perda. 

O vetor gradiente é calculado através das derivadas parciais da função e o seu valor em cada ponto indicará a direção em que o crescimento da função é máximo (ou seja, o inverso do vetor gradiente indica a direção em que está o mínimo da função).

Com a informação obtida através do vetor gradiente, pode-se determinar quais variáveis do modelo devem ser alteradas e se elas devem aumentar ou diminuir para resultar na minimização do erro.

Para entender melhor, vamos supor que o resultado do modelo seja dado por uma função $f$ de três variáveis $a, b, c$ e que o modelo tem um learning rate de $α$.

$$f(a,b,c)$$

O vetor gradiente dessa função será dado por:

$$(\frac{∂f}{∂a}, \frac{∂f}{∂b}, \frac{∂f}{∂c})$$

Com o gradiente decendente a alteração das variáveis para chegar ao erro mínimo seria feita da seguinte forma:

$$a = a - α\frac{∂f}{∂a}$$

$$b = b - α\frac{∂f}{∂b}$$

$$c = c - α\frac{∂f}{∂c}$$

Não parece complicado, mas no caso de redes neurais em que as variáveis de uma camada dependem das variáveis das camadas anteriores, o ajuste feito pelo **Gradiente Decendente** se torna mais complexo, pois requer que o cálculo das derivadas parciais da função seja propagado em todas as camadas da rede.Essa propagação é chamada de **backpropagation** e no nosso modelo será feita pela função **backward** do torch.
___
## **Referências**

https://www.geeksforgeeks.org/adam-optimizer/
https://www.geeksforgeeks.org/optimization-techniques-for-gradient-descent/
https://en.wikipedia.org/wiki/Exponential_smoothing
https://www.geeksforgeeks.org/pytorch-connection-between-lossbackward-and-optimizerstep/
https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/