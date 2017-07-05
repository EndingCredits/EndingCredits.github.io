---
layout: post
title: Set Networks - Another tool for your toolbox
---

>*Q*: What do recent teqniques for [neural machine translation](https://arxiv.org/abs/1706.03762), [relational reasoning](https://arxiv.org/abs/1612.00222), and [learning on point-clouds](https://arxiv.org/abs/1612.00593) have in common?
>
>*A*: they all use set networks!

A deep-learningomancer's toolkit is seemingly endless. Simple spells, such as batch-normalisation and skip connections are joined by arcane magic such as  (not to mention the six hundred and sixty six different species of GAN).

Needless to say, the number of different model architectures is also collosal.
However, broadly speaking, we can divide most deep learning models into one of three categories: those designed to operate on feature vectors (*e.g.* fully connected), those designed to operate on tensor-structured data such as images, or audio (*e.g.* CNNs), and those designed to work on sequential data (*e.g.* RNNs).

While this is far from a comprehensive classification of all models (doubtless large numbers of networks exist for a whole host of different esoteric input formats, not to mention the dreaded [chimera](https://arxiv.org/abs/1706.05137)), but it does seem to feature one glaring ommision, namely models designed to operate on sets.

<!-- Why sets? -->
<!-- Why this doc? -->
<!-- What this doc is -->

### The simple set network
<!-- Definition of set network: equivariance and symmetry -->
Just as CNNs operate over fixed-size vectors arranged into a grid pattern, and RNNs operate over fixed-size vectors arranged into a sequence, we are interested in models that can operate over fixed-size vectors organised a set.

Unfortunately we're working with *tensorflow*, not *setflow* so we can't simply input a set into our network. We can get around this by representing our set as a tensor: Given a set of vectors `{ e1, e2, ..., en }` we can easily turn this into a single tensor by concatenating the vectors along an extra *0*th dimension. Note (and this will be important later) that a given set can have multiple representations since we can permute elements of a set without changing it, hence any permutation along the *0*th axis of our tensor is a representation of the same set.

>If our set is given as a python list, `set = [e1, e2, ..., en]`, we can use the tersorflow operation `tensor = tf.pack(set)`. However, in practice, it's easier to do this in either python or numpy as part of the data-preparation process, and feed tensorflow a single tensor.


>An important note is how to handle sets of different numbers of elements. This is not a problem if we are only working with one set at a time as we can set the *0*th axis to be dynamically sized, but when we want to batch multiple input tensors into a single batch tensor, this can cause problems as the inputs will have different dimensions. 
>The way around this is to pad all input tensors to a given size by using dummy input elements. It's important to also use a  mask to indicate which elements are real, and which ones are dummies.

<!-- A note on masking -->

<!-- A note on masking -->

### Deep set networks

#### Other layer types


### Other techniques

#### Self attention

#### Heirarchical set networks
