# Compare Interpretability Techniques

## Dilemma: Interpretable or Powerful
Some models are intrinsically interpretable, e.g. linear models, but not very powerful. Deep neural network is more powerful but is difficult to interpret, and people often blame deep neural networks for being black box. It is obviously important that we need to make people understand these models and furthermore how they make decisions so that we can improve our models and use them in practice. Now there are a few techniques coming up to explain machine learning models. I am curious what is the difference among these techniques and how they perform on the same model. 

This project aims to compare machine learning interpretability techniques. Since this is a broad topic, I need to narrow it down. Three criteria are: 1) local explanation instead of global explanation; 2) deep convolutional neutral network; 3) image classification. Accordingly, this project uses three techniques ([DeepLIFT](https://github.com/kundajelab/deeplift), [SHAP](https://github.com/slundberg/shap), and [LIME](https://github.com/marcotcr/lime)), and two datasets ([handwritten digits](http://yann.lecun.com/exdb/mnist/), and [fashion product images](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)). 

“Interpretability is the degree to which a human can understand the cause of a decision.” — Tim Miller

## Some Results
I train a CNN model in Keras, and utilize these three techniques to explain it. Here we can see how they explain digit seven and eight. They are similar but also slightly different.
<p align="center">
<img src="https://github.com/yaliu777/Compare-Interpretability-Techniques/blob/main/images/compare_digit.png" width="700" />
</p>
I also finetune vgg16 on fashion product dataset. The following images show how they explain flip flops and casual shoes. There may be a problem of sparseness in GradientExplainer, and a problem of instability in LIME. Overall I think DeepExplainer's explanation is in correspondence with human tuition. 
<p align="center">
<img src="https://github.com/yaliu777/Compare-Interpretability-Techniques/blob/main/images/compare_flip.png" width="700" class="center" />
<img src="https://github.com/yaliu777/Compare-Interpretability-Techniques/blob/main/images/compare_shoes.png" width="700" class="center" />
</p>
Another interesting finding about DeepExplainer is that after I rotate an image 90 degree or 180 degree, its explanation changes as model's prediction changes.
<p align="center">
<img src="https://github.com/yaliu777/Compare-Interpretability-Techniques/blob/main/images/rotate.png" width="700" />
</p>
