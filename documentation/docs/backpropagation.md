# **Back-Propagation**

Como explicado no **Otimizador** em uma função de perda no formato $f(a,b,c)$, calcular a derivada da função de perda é suficiente para atualizar os parâmetros $a$, $b$ e $c$ de forma a minimizar o erro do modelo.

Entretanto, no caso da rede neural, o formato da função de perda é mais parecido com:

$$g(h(a,b), j(c,d), k(e,f))$$

Em que a função final depende de outras funções. Isso porque, cada camada, ao invés de receber diretamente parâmetros do modelo, recebem as saídas dos nós da camada anterior. Apenas a primeira camada recebe apenas parâmetros do modelo.

Dessa forma, para determinar o impacto de cada parâmetro no erro do modelo, apenas a derivada da função de perda não é suficiente, essa derivada deverá ser propagada por todas as camadas, isso é chamado **back-propagation**.

Vamos usar como exemplo a função, que seria o formato da função de perda de uma rede neural bem simples, com duas camadas.

$$g(h_1(a,b), h_2(c,d))$$

Em que as funções $h$ seriam as saídas dos nós da penúltima camada (como nesse caso essa também é a primeira camada, as funções $h$ depende apenas dos parâmetros do modelo).

Nesse caso, calcular a derivada da função de perda em relação aos parâmetros do modelo (que é o vetor gradiente) requer uma regra da cadeia: 

$$\frac{∂g}{∂a} = \frac{∂g}{∂h_1} \frac{∂h_1}{∂a}$$

$$\frac{∂g}{∂b} = \frac{∂g}{∂h_1} \frac{∂h_1}{∂b}$$

$$\frac{∂g}{∂c} = \frac{∂g}{∂h_2} \frac{∂h_2}{∂c}$$

$$\frac{∂g}{∂d} = \frac{∂g}{∂h_2} \frac{∂h_2}{∂d}$$

Portanto, a derivada teria que ser calculada em todas as camadas do modelo. Começando pela última camada, teríamos:

$$(\frac{∂g}{∂h_1}, \frac{∂g}{∂h_2})$$

Após isso, seriam calculadas as derivadas na penúltima camada:

$$(\frac{∂h_1}{∂a}, \frac{∂h_1}{∂b}, \frac{∂h_2}{∂c}, \frac{∂h_2}{∂d})$$

Só então teríamos informações suficientes para atualizar os parâmetros do modelo.
___
## **Referências**
