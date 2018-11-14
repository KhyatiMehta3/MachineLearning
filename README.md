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

## Day2 - 13/11/2018
### Understand a simple machine learning project 
So I came a across a simple machine learning project, just a couple of line of code in Python, by using scikit-learn ML library.
This code is actually from [this blog.](https://towardsdatascience.com/simple-machine-learning-model-in-python-in-5-lines-of-code-fe03d72e78c6)
   * Here the author creates a model and teaches it to predict the output of a dependent variable using 3 independent variables. 
   * He generates the inputs by iterating randomly up to a limit and assigning an output using some equation. 
   * He uses Linear Regression to teach the model. As far as I understand LR, it is used to predict an output(dependent variable), given an input (independent variable) where the output doesn't have a decided relationship with the input, i.e. its relationship is probabilistic/statistical in nature. Examples for a decided relationship between input and output can be E=mc^2, V=IR, x+my=c, and so on.
   * This is intuitively true, because why would you need to train any model, if you knew the output equation i.e. by extension, its relationship with the input? It'd be like teaching a model to tell you the output of y = a + b, given a, b. 
   * Although, this could be useful when you don't know the relationship of the inputs & outputs in your existing data.
   * Like, say you have lots of mathematical series, where there are inputs and 1 output & you don't know what equation to fit, so you get the same output given these inputs. Then LR can be conversely used to find the statistical relationship between both.
___

## Day3 - 14/11/2018
### Making a simple ML project, to predict weight of a student, given his/her height, using Linear Regression
Today, I'll be using a .txt file with the data of just 10 students' heights and weights, & train a model on that data, then check what would be the output weight, if I pass a height. 
Linear regression is a simple best line fitting technique, which will use the least mean square error reduction, to come to the best line from an infinite number of possible lines to fit the data so that there's minimum difference between the actual and predicted output.
The code is [here](https://github.com/KhyatiMehta3/MachineLearning/blob/master/LinearRegression_Simple.py)
The code is under progress and currently gets the input data from the text file [here](https://github.com/KhyatiMehta3/MachineLearning/blob/master/height_weight_lineaRegression_data.txt)



   

