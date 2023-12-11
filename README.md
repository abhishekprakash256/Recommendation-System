

### Description 

The repo is for building a recommendation system for the product database that is used in the anonymous organization. Using the Tensorflow. The requirments can be checked from requirements.txt file.







## The Knowledge for recommendation system

A recommendation system (or recommender system) is a class of machine  learning that uses data to help predict, narrow down, and find what  people are looking for among an exponentially growing number of options.

Recommender systems are trained to understand the preferences, previous  decisions, and characteristics of people and products using data  gathered about their interactions. These include impressions, clicks,  likes, and purchases. Because of their capability to predict consumer  interests and desires on a highly personalized level, recommender  systems are a favorite with content and product providers. They can  drive consumers to just about any product or service that interests  them, from books to videos to health classes to clothing.

While there are a vast number of recommender algorithms and  techniques, most fall into these broad categories: collaborative  filtering, content filtering and context filtering.

**Collaborative filtering** algorithms recommend items  (this is the filtering part) based on preference information from many  users (this is the collaborative part). This approach uses similarity of user preference behavior, given previous interactions between users  and items, recommender algorithms learn to predict future interaction.  These recommender systems build a model from a user’s past behavior,  such as items purchased previously or ratings given to those items and  similar decisions by other users. The idea is that if some people have  made similar decisions and purchases in the past, like a movie choice,  then there is a high probability they will agree on additional future  selections. For example, if a collaborative filtering recommender knows  you and another user share similar tastes in movies, it might recommend a movie to you that it knows this other user already likes.

AI-based recommender engines can analyze an individual’s purchase  behavior and detect patterns that will help provide them with the  content suggestions that will most likely match his or her interests.  This is what Google and Facebook actively apply when recommending ads,  or what Netflix does behind the scenes when recommending movies and TV  shows.

### Matrix Factorization for Recommendation

[Matrix factorization](https://developer.nvidia.com/blog/accelerate-recommender-systems-with-gpus/) (MF) techniques are the core of many popular algorithms, including word embedding and topic modeling, and have become a dominant methodology  within collaborative-filtering-based recommendation. MF can be used to  calculate the similarity in user’s ratings or interactions to provide  recommendations. In the simple user item matrix below, Ted and Carol  like movies B and C. Bob likes movie B. To recommend a movie to Bob,  matrix factorization calculates that users who liked B also liked C, so C is a possible recommendation for Bob.

![Matrix factorization (MF).](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-5.png)





## Deep Neural Network Models for Recommendation

There are [different variations](https://www.asimovinstitute.org/neural-network-zoo/) of artificial neural networks (ANNs), such as the following:

- ANNs where information is only fed forward from one layer to the next are called [feedforward neural networks](https://developer.nvidia.com/discover/artificial-neural-network).  Multilayer perceptrons (MLPs) are a type of feedforward ANN consisting  of at least three layers of nodes: an input layer, a hidden layer and an output layer. MLPs are flexible networks that can be applied to a  variety of scenarios.
- [Convolutional Neural Networks](https://blogs.nvidia.com/blog/2018/09/05/whats-the-difference-between-a-cnn-and-an-rnn/) are the image crunchers to identify objects.
- [Recurrent neural networks](https://blogs.nvidia.com/blog/2018/09/05/whats-the-difference-between-a-cnn-and-an-rnn/#:~:text=CNNs are called “feedforward” neural,can serve as feedback loops.) are the mathematical engines to parse language patterns and sequenced data.

### Neural Collaborative Filtering 

The [Neural Collaborative Filtering ](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF)(NCF) model is a neural network that provides collaborative filtering based  on user and item interactions. The model treats matrix factorization  from a non-linearity perspective. NCF TensorFlow takes in a sequence of  (user ID, item ID) pairs as inputs, then feeds them separately into a  matrix factorization step (where the embeddings are multiplied) and into a multilayer perceptron (MLP) network.

The outputs of the matrix factorization and the MLP network are then  combined and fed into a single dense layer that predicts whether the  input user is likely to interact with the input item.

![Combining matrix factorization and the MLP network outputs.](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-7.png)

### Contextual Sequence Learning

A [Recurrent neural network](https://developer.nvidia.com/discover/recurrent-neural-network) (RNN) is a class of [neural network](https://developer.nvidia.com/discover/artificialneuralnetwork) that has memory or feedback loops that allow it to better recognize  patterns in data. RNNs solve difficult tasks that deal with context and  sequences, such as natural language processing, and are also used for  contextual sequence recommendations. What distinguishes sequence  learning from other tasks is the need to use models with an active data  memory, such as [LSTMs](https://developer.nvidia.com/discover/lstm) (Long Short-Term Memory) or [GRU](https://developer.nvidia.com/discover/recurrent-neural-network) (Gated Recurrent Units) to learn temporal dependence in input data.  This memory of past input is crucial for successful sequence learning.  Transformer deep learning models, such as BERT (Bidirectional Encoder  Representations from Transformers), are an alternative to RNNs that  apply an attention technique—parsing a sentence by focusing attention on the most relevant words that come before and after it.  Transformer-based deep learning models don’t require sequential data to  be processed in order, allowing for much more parallelization and  reduced training time on GPUs than RNNs. 

![NMT components.](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-9.png)

### DLRM

[DLRM](https://developer.nvidia.com/blog/optimizing-dlrm-on-nvidia-gpus/) is a DL-based model for recommendations introduced by Facebook [research](https://arxiv.org/pdf/1906.00091.pdf). It’s designed to make use of both categorical and numerical inputs that are usually present in recommender system training data. To handle  categorical data, embedding layers map each category to a dense  representation before being fed into multilayer perceptrons (MLP).  Numerical features can be fed directly into an MLP.

At the next level, second-order interactions of different features  are computed explicitly by taking the dot product between all pairs of  embedding vectors and processed dense features. Those pairwise  interactions are fed into a top-level MLP to compute the likelihood of  interaction between a user and item pair.

![Probability of clicking on a recommendation.](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-12.png)







```
https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz
https://www.nvidia.com/en-us/glossary/data-science/recommendation-system/
```

