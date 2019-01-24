# decision-tree

README
Program: Decision Tree
Performs creation/training of a “Decision Tree” using training data and then can be used for classification of new unclassified data.

Program Description: 
Takes input one training file and one testing file in CSV format (of specific attribute, attribute values and class label). Builds a decision tree using the training data to be used for classification of the testing data. Has options to select either entropy or ginning index for information gain calculation. Also. Uses chi-square test for split test for stopping the splitting, and the confidence level of chi-square can be changed dynamically.

Running The Program:
We have provided a python script named ‘decision_tree.py’ which can be run from command line in any operating system. To run the script first make the folder with the python script ‘decision_tree.py’ as the current folder. Also, make sure that both the training file and testing file are present in the same folder.

Requirements:
Needs python version 3.6 or higher available to run the script.

Has dependency on the following python modules:

pandas ( to read csv files)
numpy (for array manipulation)
math (to calculate logarithmic values)
copy (to perform copy of variables)
csv (to write into csv files)
os ( to extract selected filenames)
sys (to read from command line arguments)

Usage:
An example command line input to run the script:

python decision_tree.py training.csv testing.csv 1 95

Here first argument ‘python’ asks to use python for compiling, ‘decision_tree.py’ is the python script name, third argument ’training.csv’ mentions the file name to be used for training purpose of the decision tree, fourth argument ‘testing.csv’ mentions the file name to be used for testing the decision tree. Fifth argument ‘1’ tells the program to use entropy for information gain calculation, the last argument ’95’ is the value for confidence level.

To run the script successfully all the four extra argument after the python script name needs to be mentioned.

Usage Options:
A generic command line input to run the script:

python script_name option1 option2 option3 option4

script_name: name of the python script (here, ‘decision_tree.py’)
option1: training file for the decision tree
option2: testing file for performing classification
option3: 1 for using ‘Entropy’ for information gain calculation, any other value for using ‘Gini-Index’ for information gain calculation.
option4: 99 for 99% confidence level of chi-square test, 95 for 95% confidence level of chi-square test. 0 for 0% confidence level of chi-square test. 

Built With:

Python version 3.6.2
