# Compare Interpretability Techniques

## Dilemma: Interpretable or Powerful
Some models are intrinsically interpretable, e.g. linear models, but not very powerful. Deep neural network is more powerful but is difficult to interpret, and people often blame deep neural networks for being black box. It is obviously important that we need to make people understand these models and furthermore how they make decisions so that we can improve our models and use them in practice. Now there are a few techniques coming up to explain machine learning models. I am curious what is the difference among these techniques and how they perform on the same model. 

This project aims to compare machine learning interpretability techniques. Since this is a broad topic, I need to narrow it down. Three criteria are: 1) local explanation instead of global explanation; 2) deep convolutional neutral network; 3) image classification. Accordingly, this project uses three techniques ([DeepLIFT](https://github.com/kundajelab/deeplift), [SHAP](https://github.com/slundberg/shap), and [LIME](https://github.com/marcotcr/lime)), and two datasets ([handwritten digits](http://yann.lecun.com/exdb/mnist/), and [fashion product images](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)). 

“Interpretability is the degree to which a human can understand the cause of a decision.” — Tim Miller

