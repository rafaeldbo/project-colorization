# **Otimizador**


___
## **Gradiente Decendente**

Na fase de treinamento, após calcular o erro do modelo pela **função de perda**, é necessário ajustar os parâmetros do modelo para **minimizar** esse erro. A técnica de gradiente descendente irá determinar como devem ser feitos esses ajustes utilizando o **vetor gradiente** da função de perda. 

O vetor gradiente é calculado através das derivadas parciais da função e o seu valor em cada ponto indicará a direção em que o crescimento da função é máximo (ou seja, o inverso do vetor gradiente indica a direção em que está o mínimo da função).

Com a informação obtida através do vetor gradiente, pode-se determinar quais variáveis do modelo devem ser alteradas e se elas devem aumentar ou diminuir para resultar na minimização do erro. 

No caso de redes neurais em que as variáveis de uma camada dependem das variáveis das camadas anteriores, o ajuste feito pelo **Gradiente Decendente** se torna mais complexo pois requer que o cálculo das derivadas parciais da função seja propagado em todas as camadas da rede. Essa propagação é chamada de **backpropagation** e no nosso modelo, será feita pela função **backward** do torch.
___
## **Referências**

https://www.geeksforgeeks.org/adam-optimizer/
https://www.geeksforgeeks.org/optimization-techniques-for-gradient-descent/
https://www.geeksforgeeks.org/pytorch-connection-between-lossbackward-and-optimizerstep/
https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/