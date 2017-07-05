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

Unfortunately we're working with *tensorflow*, not *setflow* so we can't simply input a set into our network. We can get around this by representing our set as a tensor: Given a set of vectors `{v1, v2, ..., vn}` we can easily turn this into a single tensor by concatenating the vectors along an extra *0*th dimension. Note (and this will be important later) that a given set can have multiple representations since we can permute elements of a set without changing it, hence any permutation along the *0*th axis of our tensor is a representation of the same set.

>If our set is given as a python list, `set = [v1, v2, ..., vn]`, we can use the tersorflow operation `tensor = tf.pack(set)`. However, in practice, it's easier to do this in either python or numpy as part of the data-preparation process, and feed tensorflow a single tensor.

#### Pooling

Let's assume for now that we want our model to produce a single fixed-size output vector, *i.e.* our model is a function 
<!-- f_k: X^n \rightarrow \mathbf Y, where X = \mathbf R^m} and Y = \mathbf R^{D_{OUT}} --> (in fact, we really need a sequence of functions <!-- \{ f: X^k \rightarrow \mathbf Y \}_{k \in \mathbf N} --> as we want to handle sets with different lengths k). We want the output to be invariant to different representations of the same the input set. Hence, we are looking for symmetric functions of vectors.

Fortunately some very simple candidates for these already exist, including most *tensorflow* ops beginning with `tf.reduce_`.
<!-- In fact we can devise such an function using any associative symmetric operation. -->
By reucing along the *0*th axis these operations transform an n times m dimensional tensor into a single m-dimensional vector.
For now we will only consider max pooling, *i.e.* `tf.reduce_max`, but most of the following will also apply to alternatives such as `tf.reduce_sum`, and `tf.reduce_sum`.

However, naively applying max pooling to our input tensor is unlikely to give us good results. For example, if our input is a set of points (a point-cloud), then max pooling will give us one half of the bounding box of all points. While this is useful information, it doesn't tell us anything about the structure of the points. We need to first transorm our input points where max pooling preserves more information about the structure of the set. Additionally, max pooling doesn't have any trainable parameters, so we need an additional trainable component to turn our architecture into a trainable model.

<!-- A note on masking -->
>An important note is how to handle sets of different numbers of elements. This is not a problem if we are only working with one set at a time as we can set the *0*th axis to be dynamically sized, but when we want to batch multiple input tensors into a single batch tensor, this can cause problems as the inputs will have different dimensions. 
>The way around this is to pad all input tensors to a given size by using dummy input elements. It's also useful to also use a  mask to indicate which elements are real, and which ones are dummies, to ensure the dummy elements don't iterfere with the real ones under certain operations.

#### Element embeddings

For each of our input vectors, we would like to transorm it in some way to work better with max pooling. We could do this via some static operation, but a much better alternative is to use some trainable function. A nice convenient function is a fully connceted neural network with *m* input neurons and *d* output neurons. Given a network `net(x)` and input tensor `<v1, v2, ..., vn>`, we can apply this network form a new tensor `<net(v1), net(v2), ..., net(vn)>`. We'll call such an operation 'set transformation by `net()`'.

> We can view this network set transformation as a single network with a weight-sharing scheme (in fact, it turns out that this is the same as applying a sequence of 1-dimensional convolution filters). By doing this we see that we can train this architecture using standard backpropogation and gradient descent. 
> A more comprehensive explanation (as well as some additional information on deep set networks) can be found in [this paper](https://arxiv.org/abs/1611.04500).

We can implement a linear set transormation layer in tensorflow via `layer_out = tf.map_fn(lambda x: tf. matmul(x, w) + b), layer_in)`, however, it's better (for hardware optimisation) to use a convolution operation `layer_out = tf.nn.conv1d(layer_in, [W_1], stride=1, padding="SAME") + b`. These layers can be stacked (along with activation functions) to create a full network set transformation.

Another way of looking at this is saying we would like some class of functions of the form <!-- \{ f_k: X^k \rightarrow \times \theta \mathbf Y^k \}_{k \in \mathbf N} --> where a change in permutation to the inputs is matched by a corresponding change in permutation to the outputs, *i.e.* each <!-- f_k --> is equivariant (excluding the parameter-space). This encompasses a greater space of possible functions than applying a simple transfomration to each vector, as it allows the transformation of each vector to depend on other vectors in the set. We'll call this larget category 'set operations' where set transofrmations are a subset. These will be useful later when we discuss deep set networks, and self-attention.


#### Putting everything together

We can now combine this set operation with our pooling operation to obtain a set representation. While we could use this set representation directly, personal experience suggests that it's better to feed this representation into a final fully connected network (presumably as this allows the set operation to focus on preserving as much information as possible when pooled).

<!-- A note on why max pooling -->

<!-- Experiment code & instructions -->


### Deep set networks


#### Enhanced element embeddings

<!-- While the primary purpose of set operations... -->

#### Other layer types


### Other techniques

#### Self attention

#### Heirarchical set networks
