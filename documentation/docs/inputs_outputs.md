# **Inputs e Outputs**

Queremos treinar um modelo capaz de transformar imagens em preto e branco (escala de cinza) em imagens coloridas. Dessa forma, mesmo ele sendo generativo é mais correto afirmar que nosso modelo realizará uma melhoria na imagem, ou invés de criar uma totalmente nova.

## **Input**

A imagens são conjuntos de dados constituídos de 3 dimensões: Altura, Largura e uma terceira dimensão que consiste nas camadas que formam a cor da imagem. Formatos comuns, como `PNG` ou `JPG`, possuem essa dimensão de cor no sistema `RGB`, que divide essa dimensão em três camadas: **R**ed **G**reen **B**lue. Mesmo imagens em preto e branco, quando nesses formatos, possuem essas três camadas, porém elas poderiam ser representadas por apenas uma.

Uma outra forma de representar camadas de cor é o sistema `LAB` que consiste em uma camada de **L**uminosidade e duas camadas de cor: camada **A**, que representa o equilíbrio entre o Vermelho e o Verde, e a camada **B** que representa o equilíbrio entre o Amarelo e o Azul. Utilizando esse sistema, é possível transformar imagens coloridas em imagens em preto e branco com uma única camada, para isso é necessário converter do sistema `RGB` para o `LAB` e remover suas duas camadas de cor, a **A** e a **B**.

``` python title="Trantando a Imagem de Entrada"
img_path = "./exemplo.jpg"

# lê uma imagem criando um tensor no formato (Cor, Largura, Altura)
img = read_image(img_path) 

# reorganiza as dimensões para imagem para o formato padrão 
img = img.permute(1, 2, 0) # (Largura, Altura, Cor) 

# converte do sistema RGB para o LAB 
LAB_img = from_numpy(rgb2lab(img)) 

# reorganiza as dimensões para imagem para o formato do pytorch 
LAB_img = LAB_img.permute(2, 0, 1) # (Cor, Largura, Altura)

# separa apenas a camada L
gray_layer = LAB_img[0, :, :].unsqueeze(0) 

# separa as camadas A e B
color_layers = LAB_img[1:, :, :] 
```

## **Output**

Após o processamento do modelo, ele nos devolverá duas novas camadas criadas a partir da camada **L**. Utilizaremos essas camadas como se fossem as camadas **A** e **B** faltantes junto a camada **L** original para formar a imagem colorida.

Uma imagem `LAB` não é muito utilizada, por isso devemos transformar a imagem de volta em `RGB` antes de utiliza-la. Para isso basta seguir o processo inverso de se obter a imagem LAB:

``` python title="Trantando a Saída do Modelo"
### Código omitido acima

pred_color_layers = model(gray_layer)

# concatena a camada L com as camadas AB
pred_LAB_img = torch.cat((gray_layer, pred_color_layers))

# reorganiza as dimensões para imagem para o formato padrão 
pred_LAB_img = pred_LAB_img.permute(1, 2, 0) # (Largura, Altura, Cor) 

# converte do sistema LAB para o RGB 
pred_RGB_img = lab2rgb(pred_LAB_img)
```
???- info "Código completo"
    ``` python title="Código completo"
    from torch import from_numpy, cat
    from torchvision.io import read_image
    from skimage.color import rgb2lab, lab2rgb

    img_path = "./exemplo.jpg"

    # converte a imagem de RGB para LAB
    img = read_image(img_path) 
    img = img.permute(1, 2, 0)

    LAB_img = from_numpy(rgb2lab(img)) 
    LAB_img = LAB_img.permute(2, 0, 1) 

    # separa os layers
    gray_layer = LAB_img[0, :, :].unsqueeze(0) 
    color_layers = LAB_img[1:, :, :] 

    # aplica o modelo de colorização
    pred_color_layers = model(gray_layer)

    # reune os layers
    pred_LAB_img = torch.cat((gray_layer, pred_color_layers))

    # converte a imagem de LAB para RGB
    pred_LAB_img = pred_LAB_img.permute(1, 2, 0) 
    pred_RGB_img = lab2rgb(pred_LAB_img)
    ```