# CS-433 Machine Learning - Project 1
- Robin Berguerand (258970)
- Arthur Vernet (245828)
- Pablo Guilland (246550)

### Generate a prediction
In order to generate a prediction using our model, please have the training and test sets, named respectively *train.csv* and *test.csv*, located in a folder called *data*, in the same folder as the code. Then use,
$ python run.py
A prediction, called *output.csv*, will be created in the same folder *run.py* is located.

### Code architecture
The project was developed using the following file architecture:  
*run.py* : The script outputting a prediction.  
*proj1_helpers.py* : Given helper methods (e.g. to load/output *.csv* files).  
*implementations.py* : Required implementations of the 6 basic ML methods.  
*implementations_helper.py* : Contains complementary methods needed by *implementations.py* (e.g. gradient computation).  
*costs.py* : Contains a few basic cost functions (e.g. MSE, MAE).
*helper.py* : The core code necessary to generate our model and compute the resulting prediction.

### Remarks
Note that the script will compute the model weights every time it's ran, in order to showcase its speed at generating the model and prove that the weights are computed somewhere and not the results of black magic. However, the hyper-parameters (i.e. the polynomial degrees) used in the model are hardcoded (see in *helper.py* ) but this decision is justified in our report, and motivated by the fact that computing the best hyper-parameter takes time and thus it would be irrelevant to keep this step in the code.
