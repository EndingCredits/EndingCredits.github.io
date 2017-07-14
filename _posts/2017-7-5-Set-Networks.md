---
layout: post
title: Set Networks - Another tool for your toolbox
---

This post covers a number of different techniques for designing models capable of dealing with sequences of unordered inputs.
Currently WIP, all feedback welcome.

>*Q*: What do recent techniques for [neural machine translation](https://arxiv.org/abs/1706.03762), [relational reasoning](https://arxiv.org/abs/1612.00222), and [learning on point-clouds](https://arxiv.org/abs/1612.00593) have in common?
>
>*A*: They all use set networks!

A deep-learning-o-mancer's toolkit is seemingly endless. Simple spells, such as batch-normalisation and skip connections are joined by arcane magic such as  (not to mention the six hundred and sixty six different species of GAN).
Needless to say, the number of different model architectures is also collosal.
However, broadly speaking, we can divide most commonly used deep learning models into one of three categories: those designed to operate on feature vectors (*e.g.* fully connected), those designed to operate on tensor-structured data such as images, or audio (*e.g.* CNNs), and those designed to work on sequential data (*e.g.* RNNs).

While this is far from a comprehensive classification of all models (doubtless large numbers of networks exist for a whole host of different esoteric input formats, not to mention the dreaded [chimera](https://arxiv.org/abs/1706.05137)), but it does seem to feature one glaring ommission: models designed to operate on sets.

Enter the set network!


## Introduction

<!-- Why sets?
       Appear in lots of places.
       (Seq2Seq for sets)
       Alternative to sequences.
       Multi-instance learning.
       Many places where a set network is an obvious 
       Yeilding SOTA results in certain domains.
     -->
     

#### Why this post?
<!-- Why this post?
       Used in lots of places, but no overarching consensus.
       Even closely related papers miss this link.
         Papers don't mention trying different layer types
     -->
Sometimes we want to work with input data given as sets, for example a list of possible binding sites for a protein, or a set of point in a point cloud. Even when data isn't given explicitly as a set, it can often be useful to represent it as one, *e.g.* we can specify a graph as a set of edges and vertices. But how do we feed a set into a neural network?

> In fact the problem of classifying sets has already been studied extensively via the field Multiple-Instance Learning (MIL). MIL is typically defined as a classification task where the objective is to learn a mapping from sets (known as bags) of elements to a single label. However, MIL also includes the assumption that bag labels are dependent only on some function of hidden element labels, *e.g.* a bag label is y only if it contains an element with label y. We are interested in more general problems where it may not be possible to determine the set label by looking at each element individually.

An obvious first step is to treat the set as a sequence (by assigning an arbitrary ordering) and input it into a squential model. However, the choice or ordering can have a [big impact on performance](https://arxiv.org/abs/1511.06391), and sequential models ar poor at modelling long-rage dependencies, so if two closely related elements end up at opposite ends of the sequence, their relationship may be missed. Ideally we would like a similar model where the output is independent of the order of the inputs, that is, a set network. ut what would such a model look like?

As it turns out, a variety of different researchers have already run into, and managed to solve this problem in a variety of different ways. However, in most instances there seems to be little awarness of others' proposed solutions and most approaches seem to be developed largely independently (*e.g.* [Deep Sets](https://arxiv.org/abs/1611.04500) and [Pointnet](https://arxiv.org/abs/1612.00593) both employ the same novel approach to 3D object classification on the same dataset).
Most of these approaches are largely based on the same principles (*i.e.* the deep set network we describe below), but many also feature unique innovations (as well as small tweaks and optimisations) that could also be applied to other problems.
This lack of awareness means researchers waste time reinventing the wheel, and often miss useful tricks that might help improve their approaches.
Furthermore, opportunities to use set networks are often missed: a good example might be in [matching networks](https://arxiv.org/abs/1606.04080) where a bi-directional RNN, instead of a set network, is used to encode elements of an input set.

With the speed of development of Deep Learning, it is not that surprising that these techniques have failed to land on the collective Deep Learning radar. There are a plethora of emerging ideas equally deserving of our attention.
However, I personally feel that set networks, of all their various shapes and guises, are an excellent tool for a variety of tasks, and are worthy of a place in the 'core' deep learning toolkit alongside CNNs and RNNs. Hence, I would like to take this opportunity to try and raise some awareness, and attempt to collate some of the various sources of information into a more digestible format.


#### What this post is
<!-- What this post is?
       Definition of set network
       Summary from other papers (in particular deep sets).
       Attempt to unify different ideas.
       Own notes which may be useful to others.
       Code (in python and tf) so other people can get started.
       Not an attempt to describe all possible types.
     -->

Rather than an authorative overview of set networks (which I have neither the time nor expertise to write), the reader should consider this a basic introduction to the conectps and a guide to implementing such networks in a practical setting. For more details overviews and explanations, I encourage readers to go and read the various papers referred to in this post.

Throughout this post I'll be including code snippets (in my preferred combination of python/tensorflow) demonstrating how various building blocks can be implemented. Sometime in the future, I hope to accompany this with some example esperimental code (in the meantime, [this poorly documented repository](https://github.com/EndingCredits/EmbeddingNetwork) can be referred to in an emergency).
I'll also be including some of my own notes, both practical and theoretical. These can be found in the grey textboxes.


<!-- Table of contents
       In the first couple of sections I will present what I consider to be the core deep set network, which is the basis of most (but not all) approaches. 
       In the next section I will cover some of the other techniques which have been used by researchers.
-->

<!-- Feedback, notes, and acknowledgements -->
This post is very much a work in progress, and will be continually updated for the forseable future. I have tried to make it as easy as possible to read and understand , but I encourage readers to share feedback on how I can improve this further. Any further questions and/or suggestions are more than welcome. You can contact me at awwoof \<@\> hotmail.com.

## The simple set network
<!-- Definition of set network: equivariance and symmetry -->
Just as CNNs operate over fixed-size vectors arranged into a grid pattern, and RNNs operate over fixed-size vectors arranged into a sequence, we are interested in models that can operate over fixed-size vectors organised a set.

Unfortunately we're working with *tensorflow*, not *setflow* so we can't simply input a set into our network. We can get around this by representing our set as a tensor: Given a set of vectors `{v1, v2, ..., vn}` we can easily turn this into a single tensor by concatenating the vectors along an extra *0*th dimension. Note (and this will be important later) that a given set can have multiple representations since we can permute elements of a set without changing it, hence any permutation along the *0*th axis of our tensor is a representation of the same set.

>If our set is given as a python list, `set = [v1, v2, ..., vn]`, we can use the tersorflow operation `tensor = tf.pack(set)`. However, in practice, it's easier to do this in either python or numpy as part of the data-preparation process, and feed tensorflow a single tensor.

#### Pooling

Let's assume for now that we want our model to produce a single fixed-size output vector, *i.e.* our model is a function ![`f_k: X^n \rightarrow \mathbf Y`](http://latex2png.com/output//latex_761706b5d5ccc71b57d611ff7b188243.png), where ![`X = \mathbf R^m`](http://latex2png.com/output//latex_c57aa0f15ade650827acab29ac0c2b80.png) and ![`Y = \mathbf R^{D_{OUT}}`](http://latex2png.com/output//latex_e1024b96186dc05914c43f913950abc0.png) (in fact, we really need a sequence of functions [`\{ f_k: X^k \rightarrow \mathbf Y \}_{k \in \mathbf N}`](http://latex2png.com/output//latex_13d485e1d43bd55ca4aa3b87a18175f7.png) as we want to handle sets with different lengths k). We want the output to be invariant to different representations of the same the input set. Hence, we are looking for symmetric functions of vectors.

Fortunately some very simple candidates for these already exist, including most *tensorflow* ops beginning with `tf.reduce_`.
By reducing along the 0th axis these operations transform an n by m dimensional tensor into a single m-dimensional vector.
<!-- Pooling operations are the bread of set networks, they are what enable to turn set into vector --> Pooling operations, such as these, are the bread of set networks; they enable us to map an arbitrary sized set of vectors into a single vector of the same dimension.
For now we will only consider max pooling, *i.e.* `tf.reduce_max`, but most of the following will also apply to alternatives such as `tf.reduce_sum`, and `tf.reduce_sum`.

<!--  Notes on how to compose from simple arithmentic functions.
      In fact we can devise such an function using any associative symmetric operation.
      Given an associative symmetric operation on pairs of vectors, we can form a pooling operation by applying ... -->

However, naively applying max pooling to our input tensor is unlikely to give us good results. For example, if our input is a set of points (a point-cloud), then max pooling will give us one half of the bounding box of all points. While this is useful information, it doesn't tell us anything about the structure of the points. We need to first transorm our input points where max pooling preserves more information about the structure of the set. Additionally, max pooling doesn't have any trainable parameters, so we need an additional trainable component to turn our architecture into a trainable model.

<!-- A note on masking -->
>An important note is how to handle sets of different numbers of elements. This is not a problem if we are only working with one set at a time as we can set the 0th axis to be dynamically sized, but when we want to batch multiple input tensors into a single batch tensor, this can cause problems as the inputs will have different dimensions. 
>The way around this is to pad all input tensors to a given size by using dummy input elements. It's also useful to also use a  mask to indicate which elements are real, and which ones are dummies, to ensure the dummy elements don't interfere with the real ones under certain operations.
>```python
>def batch_lists(input_lists):
>    # Takes an input list of lists (of vectors), pads each list the length of
>    #  the longest list, compiles the list into a single n x m x d array, and
>    #  returns a corresponding n x m x 1 mask.
>    max_len = 0
>    out = []; masks = []
>    for i in input_list: max_len = max(len(i),max_len)
>    for l in input_list:
>        # Zero pad output
>        out.append( np.pad(np.array(l,dtype=np.float32),
>            ((0,max_len-len(l)),(0,0)), mode='constant') )
>        # Create mask...
>        masks.append( np.pad(np.array(np.ones((len(l),1)),dtype=np.float32),
>            ((0,max_len-len(l)),(0,0)), mode='constant') )
>    return out, masks
>```

#### Element embeddings

For each of our input vectors, we would like to transorm it in some way to work better with max pooling. We could do this via some static operation, but a much better alternative is to use some trainable function. A nice convenient function is a fully connceted neural network with *m* input neurons and *d* output neurons. Given a network `net(x)` and input tensor `<v1, v2, ..., vn>`, we can apply this network form a new tensor `<net(v1), net(v2), ..., net(vn)>`. We'll call such an operation 'set transformation by `net()`'.

<!-- Embedding is the butter (bread by itself is technically functional, but not very nice) -->

> We can view this network set transformation as a single network with a weight-sharing scheme (in fact, it turns out that this is the same as applying a sequence of 1-dimensional convolution filters). By doing this we see that we can train this architecture using standard backpropogation and gradient descent. 
> A more comprehensive explanation (as well as some additional information on deep set networks) can be found in [this paper](https://arxiv.org/abs/1611.04500).

We can implement a linear set transormation layer in tensorflow via `layer_out = tf.map_fn(lambda x: tf. matmul(x, w) + b), layer_in)`, however, it's better (for hardware optimisation) to use a convolution operation. These layers can be stacked (along with activation functions) to create a full network set transformation.
<!-- `layer_out = tf.nn.conv1d(layer_in, [W_1], stride=1, padding="SAME") + b` -->

Let's create a function wrapper for a simple linear set transormation layer:
```python
def linear_set_layer(layer_size, inputs):
    """ Applies a linear transformation to each element in the input set.
    
    Args
        layer_size: Dimension to ttransform the input vectors to
        inputs: A tensor of dimensions batch_size x sequence_length x vector
          dimension representing the sequences of input vectors.
    Outputs
        output: A tensor of dimensions batch_size x sequence_length x vector
          dimension representing the sequences of transformed vectors.
    """
    
    # Get the dimension of our input vectors
    in_size = inputs.get_shape().as_list()[-1]

    # Create our variables
    w = tf.Variable(tf.random_normal((in_size, layer_size))
    b = tf.Variable(tf.zeros(layer_size))
    
    # Apply 1D convolution to apply linear filter to each element along the 2nd
    #  dimension and add the bias term
    outputs = tf.nn.conv1d(inputs, [w], stride=1, padding="SAME") + b

    return outputs
```

Another way of looking at this is saying we would like some class of functions of the form ![`\{ f_k: X^k \times \theta \rightarrow \mathbf Y^k \}_{k \in \mathbf N}`](http://latex2png.com/output//latex_21ebd3e702c6d6652329339efe60463b.png) where a change in permutation to the inputs is matched by a corresponding change in permutation to the outputs, *i.e.* each ![`f_k`](http://latex2png.com/output//latex_47240a5c231d05eabaf2eb8a0c477efa.png) is equivariant (excluding the parameter-space). This encompasses a greater space of possible functions than applying a simple transfomration to each vector, as it allows the transformation of each vector to depend on other vectors in the set. We'll call this larget category 'set operations' where set transofrmations are a subset. These will be useful later when we discuss deep set networks, and self-attention.


#### Putting everything together

We can now combine this set operation with our pooling operation to obtain a set representation. While we could use this set representation directly, personal experience suggests that it's better to feed this representation into a final fully connected network (presumably as this allows the set operation to focus on preserving as much information as possible when pooled).

<!-- A note on why max pooling 
     If we model our embedding function as a map from our element to a random vector, we can treat the output of our pooling operation as a kild of bloom filter.
     A bloom filter is...
     In fact, with max pooling we can achieve even better results than a normal bloom filter-->

<!-- Experiment code & instructions -->
<!-- Example set tranformation layer -->


## Deep set networks

While tech technique described above is surprisingly powerful, it's still rather primitive: Elements are naiveley embedded into a higher dimensional space and then pooled to get a single representation. <!-- This can (and often does) give perfectly good results on a variety of task, however it can prove brittle in certain situation, for example, on point clouds with extreme variances in scale. -->
While the set networks above may be deep in terms of a number of individual layers, they are shallow in that they only produce a single set representation. We would like networks that produce multiple sucessively more refined set representations, *i.e.* deep set networks.


#### Enhanced element embeddings

An obvious potential enhancement to our architecture would be to use some statistics about the set to help inform the transorfmation process. For example, we could first normalise our inputs by dividing by the standard deviation of the set and subtracting the mean. This is a step in the right direction, however, rather than fixed statistics and operations, it would be better if we could enable our architecture to learn these by itself.

There are many ways to do this, but for now we will look at at simple extension to our set transormfarion. For this we need a netwrok which takes two inputs, `net(x,y)`. Now, given an input tensor `<v1, v2, ..., vn>` and a set statistic `s`, we can get a new tensor `<net(v1,s), net(v2,s), ..., net(vn,s)>` which can use this set-statistic to alter the transormfation. We call this type of operation a 'contextual set transformation'. Contextual set transformations are a subset of set operations, and a superset of set transformations.

There are many different choices for `net(x,y)`, but we will consider the simple case where `net(x,y)` is a fully connected neural network with `dim(x)+dim(y)` inputs and *d* outputs, where `x` and `y` are concatenated and the resulting vector is fed into the network.

<!-- Example set tranformation layer -->
Let's alter our transformation layer code to enable it to use a context:
```python
def linear_set_layer(layer_size, inputs, context=None):
    """ Applies a linear transformation to each element in the input set.
    
    Args
        layer_size: Dimension to ttransform the input vectors to
        inputs: A tensor of dimensions batch_size x sequence_length x vector
          dimension representing the sequences of input vectors.
        Context: A tensor of dimensions batch_size x vector
          dimension representing the context for the transformation.
    Outputs
        output: A tensor of dimensions batch_size x sequence_length x vector
          dimension representing the sequences of transformed vectors.
    """
    
    # Get the dimension of our input vectors
    in_size = inputs.get_shape().as_list()[-1]

    # Create our variables
    w = tf.Variable(tf.random_normal((in_size, layer_size))
    b = tf.Variable(tf.zeros(layer_size))
    
    # Apply 1D convolution to apply linear filter to each element along the 2nd
    #  dimension and add the bias term
    outputs = tf.nn.conv1d(inputs, [w], stride=1, padding="SAME") + b
    
    # Apply the context if it exists
    if context is not None:
        # Unfortunately tf doesn't support broadcasting via concat, but we can
        #  simply add the transformed context to get the same effect
        context = tf.expand_dims(context, axis=1)
        context_size = context.get_shape().as_list()[-1]
        w_c = tf.Variable(tf.random_normal((context_size, layer_size))
        cont_transformed = tf.nn.conv1d(context, [w_c], stride=1, padding="SAME")
        outputs += cont_transformed
        
    return outputs
```

But where can we get this set statistic from? We could use some generic statistic of the set (*e.g.* mean, or standard deviation), but we've already discussed a general method for representing sets with vectors: via set networks! 

> While the set operations are useful for creating better embeddings for pooling, there are many cases where individual element embeddings can used by themselves. For example, for semantic segmentation, the embedding of each element could correspond to the networks predicted class of that element. For simple set networks, these individual element embeddings are not too interesting by themselves, as they were produced without additional information from other elements in the set, however for deep set networks they are a lot more useful. 


#### Building deep set networks

Now that we have methods for using contexts (contextual transformations), and methods for generating contexts (set networks) we can put these together to build deep set networks.

A simple example of such a network is of the form:
```python
def really_simple_network(inputs, mask):
    layer_1 = tf.nn.relu(linear_set_layer(64, inputs))
    cont_1 = tf.reduce_max(layer_1 * mask, axis=1) # Mask out dummy vectors
    layer_2 = tf.nn.relu(linear_set_layer(64, layer_1, context=cont_1))
    cont_2 = tf.reduce_max(layer_2 * mask, axis=1)
    return cont_2
```
This simply applies an initial set network to obtain a context, which it then uses, along side the the transformed inputs (*i.e.* `layer_1`). for a second set network on the inputs.

In practice, we probably want to add a few more layers to this architecture. We could do this by adding extra layers on top in the same fashion, but we can also make the two individual stages deeper by turning them into multi-layer networks. We can do this by adding extra linear layers in-between:
```python
def really_simple_network(inputs, mask):
    stage_1_hidden = tf.nn.relu(linear_set_layer(64, inputs))
    stage_1_out = tf.nn.relu(linear_set_layer(64, stage_1_hidden))
    cont_1 = tf.reduce_max(layer_1 * mask, axis=1) # Mask out dummy vectors
    
    stage_2_hidden = tf.nn.relu(linear_set_layer(64, stage_1_out, context=cont_1))
    stage_2_out = tf.nn.relu(linear_set_layer(64, stage_2_hidden))
    cont_2 = tf.reduce_max(stage_2_out * mask, axis=1)
    return cont_2
```
<!-- Defn of stages -->
To avoid confusion, rather than call the individual component set networks networks, we will call them stages. Broadly speaking, a stage is a combination of layers where the lowest layer takes a context, but the layers above it (up to the next stage) do not.

<!-- Notes on architecture styles -->
Already we have many different options for building architectures, even if we fix the number of layers and layer sizes. For example, we can choose to either have many simple stages, or fewer deeper stage. Since each context layer takes two inputs, we can choose to take one of these inputs from an earlier stage of the network, for example, we could have replaced `layer_2 = tf.nn.relu(linear_set_layer(64, layer_1, context=cont_1))` with `layer_2 = tf.nn.relu(linear_set_layer(64, inputs, context=cont_1))` in our `really_simple_network`. In my own experiments I found that taking the element transformation from an earlier stage in this way can be beneficial<!-- as shown empirically -->, possibly since it allows the network to separate the job of finding a good initial context from transforming the elements.

<!-- Experiment code & instructions -->



## The Toolbox

#### Other layer types

This section will cover two alternatives to the general set transofmration layer: the set layer used by [Ravanbakhsh et al.](https://arxiv.org/abs/1611.04500) and the T-net of [Qi et al.](https://arxiv.org/abs/1612.00593).


#### Self attention

Recently Google published a paper titled [Attention Is All You Need](https://arxiv.org/abs/1706.03762) demonstrating a novel method for machine translation that demonstrated state-of-the-art performance on an English-German translation task. What was interesting about their approach is that they completely forwent any kind of sequential model, such as an RNN (this coming not long after facebook announced a [CNN model for machine translation](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/)). In fact, it turns out that the model Google used (the transformer network) is actually a kind of set network (with one caveat).

Unlike the set networks described here, the transformer network doesn't use any kind of global pooling. Instead it relies on a mechanism called 'self attention'. Rather than generating a global context, which is applied to all elements, each element produces its own context based on other elements in the set. This is done by using an attention mechanism, where each element controls how strongly other elements in the set contribute to its own context. Attention mechanism are often used in sequence-to-sequence tasks where the output elements are allowed to attend to the input elements. What transformer net does is also allow input, and output elements to attend to themselves.

This is done as follows:
1.) Each element produces a key, and a value which are fixed sized vectors.
2.) Each element also produces a query, which is a fixed sized vector the same size as the key.
3.) For each pair of elements `x_i` and `x_j` we generate a weight `w_i_j` by taking the dot product of the query vector `x_i_q` of `x_i` and the key vector `x_j_k` of `x_j` (and then passing it through a softmax). This value `w_i_j` corresponds to how much the element `x_i` 'pays attention' to `x_j`.
4.) For each element `x_i`, we produce a context `c_i` by multiplying the values of all elements by the corresponding weight `w_i_j`, and summing them together, *i.e.* `c_i = sum(w_i_j * x_j_v)`.

<!-- Code -->

<!-- Max pooiling as global attention -->
In many ways, self attention can be considered an extension of global pooling. In fact, if we use sum pooling, then this is equivalent to using self-attention with all attended weights set to 1.

#### Heirarchical set networks

This section will cover applying set-networks to sets of sets (by partitioning sets) as usied in [Pointnet++](https://arxiv.org/abs/1706.02413) and other places.


### Misc

#### List of papers
* Neural Statistician

#### To-do list / wishlist
* Empirical study on different tools for different tasks
* Note on activation functions and optimiser type (adamax)
* Notes on dealing with sets which consist of vectors of different lengths.
