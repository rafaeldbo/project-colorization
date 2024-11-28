# **Contexto**

Iremos desenvolver um modelo generativo para colorização de imagens. Como já é esperado, isso significa que o modelo finalizado terá como entrada uma imagem preto e branco e retornará uma imagem colorida. Entretanto, teremos mais uma entrada além da imagem em preto e branco, uma categoria.

Alguns padrões são facilmente reconhecidos como por exemplo, ambientes externos normalmente terão mais verde devido a presença de plantas ou azul devido ao céu, enquanto ambientes internos normalmente terão cores mais neutras. Adicionando a categoria como entrada, permitiremos que o modelo indentifique esses padrões mais específicos e os levem em conta na hora de colorir uma nova imagem.

Com base no dataset [Image Colorization Dataset](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset) definimos as categorias como:

- Ambientes externos
- Ambientes internos
- Objetos
- Pessoas
- Animais
- Veículos
- Alimentos 
