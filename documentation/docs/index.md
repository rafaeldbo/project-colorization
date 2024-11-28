# ECNN - Embedded Convolutional Neural Network

Essa é a parte teórica do Tutorial de construção de um modelo de colorização de imagens classificadas. Esse modelo deverá receber uma imagem em preto e branco (escala de cinza) e, utilizando o `tema da imagem`, tentará colorir essa imagem.

A parte prática desse tutorial está disponível [neste notebook](https://drive.google.com/uc?export=download&id=1tYX3coYY3NK9hlyUvRXODeG5-6EStecG).

Os códigos presentes no tutorial foram feita utilizando a biblioteca [PyTorch](https://github.com/pytorch/pytorch). O treinamento poderá ser feito utilizando o conjunto de treinamento do dataset [Image Colorization Dataset](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset) e a classificação manual dessas imagens feita por nós, disponivel em: [Categories](https://drive.google.com/uc?export=download&id=115OMNGbthQ5CFmnvPUlxYZ-_Y1CNxI9b).

As seguintes categorias foram usadas na classificação das imagens: `pessoas`, `alimentos`, `animais`, `veiculos`, `ambientes externos`, `ambientes internos` e `objetos`.

!!! warning "Alerta"
    É extremamente recomendado que a construção final do código presente no **Notebbok** só seja feita após a leitura completa desta parte teórica.

## **Créditos**
### **Desenvolvedores**
* Beatriz Rodrigues de Freitas
* Carlos Eduardo Porciuncula Yamada
* Rafael Dourado Bastos de Oliveira

### **Orientação**
* Fábio José Ayres