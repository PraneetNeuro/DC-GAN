# Deep Convolutional Generative Adversarial Networks
A python package that'll help you train DCGAN models with your own image based data.

# Overview of GAN 
The GAN involves two neural networks, a generator neural network and a discriminator neural network. 
GANs can be used to generate new data based on the existing training data which is highly effective in case of tasks like data augmentation.
GAN's generally fall under the unsupervised learning algorithms which doesn't required labelled data. 
But GAN's like Conditional GAN or StackGAN require labelled data because we are able to condition the output of the GAN based on an input.
Training a GAN involves training two neural networks, the generator and discriminator as previously mentioned. 
The generator neural network is responsible for generating new data that seems to be real for the discriminator. (Basically, the generator needs to fool the discriminator into thinking that it has generated real data that belongs to the training dataset that the neural networks were trained with)
The discriminator neural network is just a classifier neural network which predicts whether the data generated by the generator neural network is real or fake.
This form of training is based on the AI principle called "minimax" where there are two agents trying to minimize / maximize their opposing objective functions.
The generator neural network starts off by generating random noise which is meaningless but starts to generate convincing data as the discriminator penalises the generator with high loss when it receives data which doesn't seem real.
Once the generator neural network is able to generate new meaningful data, there is no use for the discriminator neural network. In the case of DCGAN, we'll be using a Convolutional Neural Network as the discriminator and a neural network with fractional convolutional layers as the generator.
The input to the generator is just randomly data of a particular shape.

# Requirements
1. Python 3.6
2. Tensorflow 2.3
3. opencv-python
4. matplotlib

# Usage
```python
import src

GAN('<PATH OF THE TRAINING DATASET>')
```
