---
layout: post
title: Vectorizing Gradient Computation
excerpt: A post detaling the process of computing gradients for backpropagation and on vectorizing the calculation.
mathjax: true
---

Finally I have pushed myself out of laziness to start my blog with the first post. So as to keep the inertia low, and to ease myself into the process of writing, I thought I will keep the post simple. Last semester, I worked through the lectures, notes, and _mainly_ the assignments of the famous Stanford's [CS231n CNN for Visual Recognition](http://cs231n.stanford.edu/). This course was my first entry point to Deep Learning. While doing the assignments, among other things, I had to implement multi-class SVM, Softmax,  FeedForward networks, Convolutional Networks, and Recurrent Neural Networks. The assignments were structured in a way to encourage a modular approach in building neural nets. So for each network, we build a forward pass layer and a corresponding backward pass layer (which will be two separate Python functions). In this process, one is forced to get a pen and paper and workout the math by calculating derivates/gradients for backward pass (which I think is a very good practice for beginners). Also, for obvious efficiency reasons, the code has to be vectorized. In this post I want to talk about the process of calculating the gradients and on how to vectorize it.

Here is an example MathJax inline rendering \\( 1/x^{2} \\), and here is a block rendering: 
\\[ \frac{1}{n^{2}} \\]