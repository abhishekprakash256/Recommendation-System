## The Knowledge for recommendation system

A recommendation system (or recommender system) is a class of machine  learning that uses data to help predict, narrow down, and find what  people are looking for among an exponentially growing number of options.

Recommender systems are trained to understand the preferences, previous  decisions, and characteristics of people and products using data  gathered about their interactions. These include impressions, clicks,  likes, and purchases. Because of their capability to predict consumer  interests and desires on a highly personalized level, recommender  systems are a favorite with content and product providers. They can  drive consumers to just about any product or service that interests  them, from books to videos to health classes to clothing.

While there are a vast number of recommender algorithms and  techniques, most fall into these broad categories: collaborative  filtering, content filtering and context filtering.

**Collaborative filtering** algorithms recommend items  (this is the filtering part) based on preference information from many  users (this is the collaborative part). This approach uses similarity of user preference behavior, given previous interactions between users  and items, recommender algorithms learn to predict future interaction.  These recommender systems build a model from a user’s past behavior,  such as items purchased previously or ratings given to those items and  similar decisions by other users. The idea is that if some people have  made similar decisions and purchases in the past, like a movie choice,  then there is a high probability they will agree on additional future  selections. For example, if a collaborative filtering recommender knows  you and another user share similar tastes in movies, it might recommend a movie to you that it knows this other user already likes.

AI-based recommender engines can analyze an individual’s purchase  behavior and detect patterns that will help provide them with the  content suggestions that will most likely match his or her interests.  This is what Google and Facebook actively apply when recommending ads,  or what Netflix does behind the scenes when recommending movies and TV  shows.

### Matrix Factorization for Recommendation

[Matrix factorization](https://developer.nvidia.com/blog/accelerate-recommender-systems-with-gpus/) (MF) techniques are the core of many popular algorithms, including word embedding and topic modeling, and have become a dominant methodology  within collaborative-filtering-based recommendation. MF can be used to  calculate the similarity in user’s ratings or interactions to provide  recommendations. In the simple user item matrix below, Ted and Carol  like movies B and C. Bob likes movie B. To recommend a movie to Bob,  matrix factorization calculates that users who liked B also liked C, so C is a possible recommendation for Bob.

![Matrix factorization (MF).](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-5.png)









- The system will be designed using tensorflow
- Model fitting via keras
- The stages consist of retrival, ranking, Post-ranking 
- The model is based on the tower model 





```
https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz
https://www.nvidia.com/en-us/glossary/data-science/recommendation-system/
```

