# An implementation of ID3 decision tree algorithm using
# both entropy and gini index for information gain calculation
#
# Also, has chi-square statistical test based splitt stop checking
# Please refer to the readme file for further information regarding 
# using this python script

# Python Version 3.6.2
# Developed by Farhan Asif Chowdhury (fasifchowdhury@unm.edu) and
# Md Amanul Hasan (amanulhasan@unm.edu)

# Code begins
# import necesssary python mofdules

import sys
import numpy as np
import math
import copy
import csv
import os

# parsing input argument for training file, testing file name
# and info_option (whether to use entropy or ginin indiex for information gain)
# and confidence level for chi-square test

training_file=sys.argv[1]
testing_file=sys.argv[2]

info_option=int(sys.argv[3])
confidence_level=int(sys.argv[4])

if (confidence_level==99):
    critical_value=16.812
if (confidence_level==95):
    critical_value=12.592 
if (confidence_level==0):
    critical_value=0.872
    
    
# reading training data 
data_seq=[]
data_label=[]
data_id=[]
with open('training.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data_id.append(row[0])
        data_seq.append(row[1])
        data_label.append(row[2])


# declaring noise values to search for in the data sequence
noise=['D','N','S','R']


# removing data sequences,data_id, data_label that have noise in any of their attribute values
data_id_N=[]
data_label_N=[]

data_seq_array= np.empty((1,60))

for i,seq in enumerate(data_seq):
    cur_seq=data_seq[i]
    if not [L for L in noise if L in cur_seq]:
        cur_seq=list(cur_seq)
        pp=np.array(cur_seq)
        pp=pp.reshape(1,-1)
        data_seq_array=np.append(data_seq_array,pp,axis=0)
        data_id_N.append(data_id[i])
        data_label_N.append(data_label[i])


# copying noise free data sequence values into a numpy array
set_of_sample=data_seq_array[1:][:]
set_of_label=np.array(data_label_N)
data_id_N=np.array(data_id_N)


# set of attribute
set_of_attribute=list(range(0,60)) 

# possible set of attribute value 
set_of_decision=['A','G','T','C']

# possible set of sequence labels
set_of_class=['N','EI','IE']


# function to get split attribute from a set of attributes
def get_attribute(set_of_sample, set_of_attribute,set_of_decision,set_of_class,set_of_label):

    max_gain=0
    max_gain_attribute=[]
    
    # information before splitting
    parent_gain=get_info(set_of_label,set_of_class)

    # running a for loop through all current possible attribute to calculate corresponding information gain
    for cur_attribute in set_of_attribute:
        
        # calling function to calculate information gain for current attribute
        cur_gain=parent_gain-get_total_gain(set_of_sample,cur_attribute,set_of_decision,set_of_class,set_of_label)

        # if the curreent gain is greater than all previous gain, current gain becomes the maximum
        if cur_gain>=max_gain:
            max_gain=cur_gain
            
            # attribute for which the maximum gain occurs
            max_gain_attribute=cur_attribute

    return max_gain_attribute


# function to calculate information gain for a particular selected attribute
def get_total_gain(set_of_sample,cur_attribute,set_of_decision,set_of_class,set_of_label):
    
    total_info=0
    
    total_no_of_sample=len(set_of_sample)
    
    cur_attribute_sample= set_of_sample[:,cur_attribute]

    cur_label_sample= set_of_label

    # running a for loop to calculate information for all the possible values of the selected attribute
    for cur_decision in set_of_decision:

        cur_decision_true_sample_ind=[]
        count=0
        
        
        for i, x in enumerate (cur_attribute_sample):

            count=count+1

            if x == cur_decision:
                cur_decision_true_sample_ind.append(i)


        cur_decision_true_number=len(cur_decision_true_sample_ind)  
        
        cur_weight=cur_decision_true_number/total_no_of_sample
        
        cur_attribute_cur_decision_label=cur_label_sample[cur_decision_true_sample_ind]
        
        # calling function to calculate information for on particular attribute value
        cur_info=cur_weight*get_info(cur_attribute_cur_decision_label,set_of_class)
        
        total_info=total_info+cur_info
        
    return total_info
    
    
# function to calculate information
def get_info(cur_attribute_cur_decision_label,set_of_class):
    
    
    total_no_of_sample=len(cur_attribute_cur_decision_label)+0.0001
    total_info=0
    
    # running a for loop to calculate probability for each class occurance
    for cur_class in set_of_class:
        
        count=0
        for x in cur_attribute_cur_decision_label:
            #print(x)
            if x == cur_class:
                count=count+1
        
        # probabilty of occurance of a particular class
        cur_probability=(count/total_no_of_sample)+0.0001
        
        # info_option==1 for entropy method
        if (info_option==1):
            cur_info = -cur_probability * math.log2(cur_probability)  
            total_info=total_info+cur_info
         
        # info_option==2 for gini index method
        else:
            cur_info = cur_probability * cur_probability  
            total_info=total_info+cur_info
            
            
    if (info_option==1):
        return total_info
    else:
        return 1-total_info
        

#  the main decision tree class which starts the decision tree from a root node
#  for a given training set data; also has a classification method which uses 
#  the created decision tree from training data to do classification on a new tesing data

class decision_tree():
    
    print('INSIDE DECISION TREE. TRAINING HAS BEGUN.')
    
    # class intstance initialization method of decision tree class
    def __init__ (self,set_of_sample,set_of_attribute,set_of_decision,set_of_class,set_of_label):
        
        # stroing instance specific values
        self.set_of_sample=set_of_sample
        self.set_of_attribute=set_of_attribute
        self.set_of_label=set_of_label
        
        self.set_of_decision=set_of_decision
        self.set_of_class=set_of_class
        
        parent=0
        
        # creating the root node
        self.root_node=tree_node(parent,attribute_value=None,is_leaf=False,label=None)
        
        # creating a subtree from the root node
        self.root_node.create_subtree(self.set_of_sample,self.set_of_attribute,self.set_of_decision,self.set_of_class,self.set_of_label)
    
    
    # method to perform classification of test data using the tree built from the training data
    def do_classification (self,all_test_sample):

        all_test_label = []
        
        # running a for loop through the test data sequence and performing the classification one at a time
        for i,cur_test_sample in enumerate(all_test_sample):
            
            cur_test_label=self.root_node.get_class(cur_test_sample)
            
            # Appending the classification result
            all_test_label.append(cur_test_label)
        return all_test_label  


# The tree node class to be used as the data structure to store tree node
#  information; for example node label if it is a leaf node, whether the node is a leaf node or
# the node splits into more children node, what are the children node
class tree_node():
    
    tree_count=0
    node_count=0
    
    # class instance initialization method of tree_node class
    def __init__(self,parent,attribute_value,is_leaf,label):
        
        
        # storing instance specific vaues
        tree_node.node_count=tree_node.node_count+1
        
        self.parent=parent
        self.next_parent=self.parent+1
        
        self.children=[]
        self.split_attribute=[]

        self.attribute_value=attribute_value
        self.is_leaf=is_leaf
        self.label=label
        
    # method to visulaize the tree node and their ralation,values
    def pre_order(self):
        
        if self.is_leaf:
            print ('\n')
            print(self.split_attribute)
            print(self.label)
        
        else:
            for child in self.children:
                child.pre_order()
     
    # method to calcualte chi-square value for given splitting attribute at a given node         
    def get_chi(self,set_of_sample,set_of_label,set_of_decision,set_of_class):
        
        
        
        total_chi_val=0
        total_sample_no=len(set_of_sample)+0.0001
        matrix=np.zeros(shape=(len(set_of_class)+1,len(set_of_decision)+1),dtype=int)
        
        
        for i,x in enumerate(set_of_class):
                     
            for j,y in enumerate(set_of_decision):
                
                temp_ind=[p for p,q in enumerate(set_of_label) if q==x]
                temp_sample=set_of_sample[temp_ind]
                matrix[i,j]=len([[w for w in temp_sample if w==y]])
            
            matrix[i,j+1]=np.sum(matrix[i,:])
            
        for j,y in enumerate(set_of_decision):
            matrix[len(set_of_class),j]=np.sum(matrix[:,j])
        
        for i in range(0,len(set_of_class)):
            for j in range(0,len(set_of_decision)):
                
                N_real=matrix[i,j]
                N_expected=(matrix[len(set_of_class),j]*(matrix[i,len(set_of_decision)]/total_sample_no))+0.0001
                cur_chi=(math.pow(N_real-N_expected,2))/N_expected
                
                total_chi_val=total_chi_val+cur_chi
        
        
        return total_chi_val
    
    # method to decide whether to stop splitting of the current node based on chi-square
    # value for a selected confidence level
    def stop_split(self,chi_val,critical_value):
        stop_prune_val=True
        
        if chi_val>critical_value:
            stop_prune_val=False    
        
        return stop_prune_val
    
    
    # method to get the label if current node is a leaf node or to recursively call the children node 
    def get_class(self,cur_test_sample):
        
        if  self.label:
            return self.label
        
        
        for child in self.children:

            if (child.attribute_value == cur_test_sample[self.split_attribute]):

                return child.get_class(cur_test_sample)
        
        return self.label
        
    
    # method to determine if all the current data sequence has same label and if yes, to return the label
    def all_same_label(self,set_of_label):
        label=[]
        if (len(set(set_of_label))==1):
            label=set_of_label[0]
        

        return label
    
    # method to calculate the class which occurs the most among the current sequence labels
    def get_maximum_label(self,set_of_label):
        
        label=[]
        unique_element=list(set(set_of_label))
        
        total_count=[]
        for i in unique_element:
            count=0
            for j in set_of_label:
                if i == j:
                    count=count+1;
            total_count=np.append(total_count,count)
        
        label=unique_element[np.argmax(total_count)]
        

        return label
        
    # method to split the sequence data and their label based on their attribute values
    def get_sample_for_split_attribute(self,set_of_sample,set_of_label,attribute_value,pack):
        
        a=np.empty(0)
        b=np.empty(0)
        attribute_matched_ind=[]
        for i, x in enumerate (pack):
            
            if x == attribute_value:
                attribute_matched_ind.append(i)

        keep_size=len(attribute_matched_ind)
        if (keep_size>0):
            a=set_of_sample[attribute_matched_ind]
            a=a.reshape(keep_size,-1)
            b=set_of_label[attribute_matched_ind]
            
        return a,b
                    
    # method to create subtree from a given node 
    def create_subtree(self,set_of_sample,set_of_attribute,set_of_decision,set_of_class,set_of_label): 
        
        tree_node.tree_count=tree_node.tree_count+1
        
        # checking if all the curent sequences set has same label
        temp_check=self.all_same_label(set_of_label)
        
        # if all the sample has same label, label the current node as
        # leaf node, give it the corresponding label, and stop further splitting and 
        # creating subtree from the node
        if  temp_check:
            self.label=temp_check
            self.is_leaf=True

            return
        # checking if there is any attribute left to slest as splitting attribute
        # if no attribute is left, mark the current node as a leaf node, give it a
        # class label which occurs the most among the current set of sequences and 
        # stop further splitting and creating subtree from the node
        if not set_of_attribute:
            self.label=self.get_maximum_label(set_of_label)
            self.is_leaf=True

            return
        
        # calling a function to select the splitting attribute which gives the maximum information gain
        self.split_attribute=get_attribute(set_of_sample,set_of_attribute,set_of_decision,set_of_class,set_of_label)
        
        # calculate chi-square value for the selected splitting attriute
        chi_val=self.get_chi(set_of_sample[:,self.split_attribute],set_of_label,set_of_decision,set_of_class)
        

        
        # for the calculated chi-square and confidence level, check whether to stop the splitting
        # and mark the node as a leaf node
        
        # if chi-square value is less than the corresponding confidence value stop the splitting
        # and mark the current node as a leaf node, give it a
        # class label which occurs the most among the current set of sequences and 
        # stop further splitting and creating subtree from the node
        
        if self.stop_split(chi_val,critical_value):
            
            self.label=self.get_maximum_label(set_of_label)
            self.is_leaf=True

            return
        
        # if splitting is to be done using the selected attribute,
        # remove the selected attribute from the set of attributes
        new_set_of_attribute=copy.deepcopy(set_of_attribute)
        new_set_of_attribute.remove(self.split_attribute)

        
        # for all the possible values of the the selected attribute create a new child node
        # of the current node
        for attribute_value in set_of_decision:
            
            # separate the current set of data sequence and their label based on their attribute values 
            # of the selected attribute
            new_set_of_sample,new_set_of_label=self.get_sample_for_split_attribute(set_of_sample,set_of_label,attribute_value,set_of_sample[:,self.split_attribute])
            
            # for the current attribute value the check if number of matched datasequence greater than zero
            # if there is data sequences that matches the current attribute value, create a new child node
            # and recursively create asubtree from that node
            if  new_set_of_sample.size:
                new_children=tree_node(self.next_parent,attribute_value,is_leaf=False,label=None)
            
                self.children=np.append(self.children,new_children)
                new_children.create_subtree(new_set_of_sample,new_set_of_attribute,set_of_decision,set_of_class,new_set_of_label)
              
            # if there is no data sequences that matches current attribute value create a child node,
            # mark it as a leaf node, give it the label which occurs the mos in it parent's set of data
            # sequence label
            else:
                temp_label=self.get_maximum_label(set_of_label)

                new_children=tree_node(self.next_parent,attribute_value,is_leaf=True,label=temp_label)
                self.children=np.append(self.children,new_children)
            
        return
                

# creating the decision tree from the training data set
my_decision_tree=decision_tree(set_of_sample,set_of_attribute,set_of_decision,set_of_class,set_of_label) 



# testing phase
print('\n')
print ('Training Done.')


# reading testing data 
data_seq=[]
data_id=[]

with open(testing_file) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data_id.append(row[0])
        data_seq.append(row[1])


# copying data sequence values into a numpy array

data_seq_array= np.empty((1,60))

for i,seq in enumerate(data_seq):
    cur_seq=data_seq[i]
    cur_seq=list(cur_seq)
    pp=np.array(cur_seq)
    pp=pp.reshape(1,-1)
    data_seq_array=np.append(data_seq_array,pp,axis=0)


data_id_test=np.array(data_id)
all_test_sample=data_seq_array[1:][:]



# doing the classification on the test data using the decision tree built using training data
all_test_label=my_decision_tree.do_classification(all_test_sample)
    

# appending the test data sequence id and data sequence class label in a list
table=[]
pp=str('id')
cc=str('class')
table.append([pp,cc]) 
for i,x in enumerate(data_id_test):
    y=all_test_label[i]
    table.append([x,y])

output_file=os.path.splitext(os.path.basename(testing_file))[0]

if (info_option==1):
    output_file=output_file+'_entropy'
else:
    output_file=output_file+'_gini'
    
if (confidence_level==99):
    output_file=output_file+'_99_result.csv'
if (confidence_level==95):
    output_file=output_file+'_95_result.csv'
if (confidence_level==0):
    output_file=output_file+'_0_result.csv'

# creating a csv file of data sequence id and their corresponding  class label 
with open(output_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in table:
        writer.writerow(val)

print('\n')
print ('Testing file classification done.')
print('\n')
print ('Test result written in a CSV file named {}'.format(output_file))