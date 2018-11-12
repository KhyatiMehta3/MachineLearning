## 100 days of working on Machine Learning - Challenge!
____
Will upload some project code and text content for the next 100 days in this repository.
## Day1 : 12/11/2018.
### Bunch of things to do here :

1. Decide on the problem to solve & check what it category it comes under -
    * Supervised learning -
        * Classification - Example Classify images as cat           or dog.
        * Regression - Example Algorithmic Trading.
    * Unsupervised learning -
        * Clustering - Example Gene Sequence analysis.
        
2. Create a dataset or download one. General rule is, the larger the dataset, the better.

3. Decide which ML library to use - Keras, Scikit-learn, TensorFlow, Theano, etc.

4. Decide which algorithm to use - 
    * This depends on which type of problem is being solved.
    * There's no best algorithm for anyone problem type. Based a little on trial and error, the algorithm can be selected.
    * For Classification problems we have -
        1. Support Vector Machine.
        2. Discriminant analysis.
        3. Naive Bayes.
        4. Nearest Neighbor.
    * For Regression problems we have -
        1. Linear Regression.
        2. SVR, GPR.
        3. Ensemble Methods.
        4. Decision trees.
        5. Neural Nets.
    * For Unsupervise learning we have -
        1. K-means, K-medoids.
        2. Gaussian Mixture.
        3. Neural nets.
        4. Hidden Markov Models.
        
5. Use one of these algorithms which suits the data best and get the output.

#### Model improvement & debugging methods :

* Data can be split into training set and test set - this allows you to test your model before deploying it anywhere.
* Plot your data to see visually if any points are out of the general pattern. Chances are, they're mere noise data. If the model tries to fit these in, it might end up predicting noise even when it gets legit input.
* Try out different models and plot the confusion matrix - Check which model has lower amount of accuracy in guessing the output. This confusion matrix helps recognize the shortcomings of a given algorithm for that problem.
* If your model is too confused between certain similar data, then your model is probably overly generalised. This means it requires a little more complexity brought into it  - Try adding more data.
* If your model is too complex, it tends to overfit everything and confidently can give wrong prediction - In this case, generalize your model, make it simpler.

