# Import the required librray
import sys
import json
import numpy as np
import re
import csv
eps = np.finfo(float).eps
from numpy import log2 as log

#commandline param
path= sys.argv[1] #train
path1= sys.argv[2] #test
max_depth = int(sys.argv[3]) 
path2 =sys.argv[4] #train out
path3= sys.argv[5] #test out
path4= sys.argv[6] # metrics


# variables
datasets =[]
copy_train=[]
test=[]
t_dat=[]
col_names=[]
class_labels =[]
unique_labels =[]
class_indx=0
entropy =0
error_rate=0
depth=4
node_cntr=[]
outpt_h_tr=[]
outpt_h_te=[]

#read file for operation 
with open(path, 'r') as input_csv:            
            reader = csv.reader(input_csv)            
            datasets = [r for r in reader]
            
with open(path1, 'r') as input_csv:            
            reader = csv.reader(input_csv)            
            t_dat = [r for r in reader]
          
 #prepare data for test    
reader = csv.DictReader(open(path1, 'r'))
for line in reader:
    test.append(line)
    
reader1 = csv.DictReader(open(path, 'r'))
for line in reader1:
    copy_train.append(line)


            
#seperate column header in a list
col_names = datasets[0]
datasets.pop(0)
t_dat.pop(0)

#get index of class label in data
if(len(datasets) > 0):
    # class label exist at tis index
    class_indx =len(datasets[0])-1 
 
#check
if(len(col_names) < max_depth):
    max_depth = max_depth -(max_depth-len(col_names)) 
    
#Extract unique value for a column
def extract_uniquevalue(data,col_idx):
    uniq_v=[]
    clas_val=[]
    #extract the unique class label
    for i in range (0,len(data)):
        label = data[i][col_idx]
        clas_val.append(label)
        # add unique label
        if label not in uniq_v: 
            uniq_v.append(label)
    return uniq_v,clas_val

#This will calculate the entropy of node
def find_entropy_base(data):
    #get unique labels for class label
    entropy=0
    unique_labels,cls_val = extract_uniquevalue(data,class_indx)
    for u in unique_labels:
        fraction = float(cls_val.count(u))/  float(len(data))
        entropy += -fraction*np.log2(fraction)
    return entropy


#Find entropy of the attributes
def find_entropy_attrb(data,attrb_idx):
    
    class_unq_vals,cls_data = extract_uniquevalue(data,class_indx)#get all the unique value for class Column    
    
    attrib_uniq_vals, attrib_data=extract_uniquevalue(data,attrb_idx)#get all the unique value for attribute Column
    
    entrpy_1=0
    for attrib in attrib_uniq_vals:
        entropy = 0
        for class_var in class_unq_vals: 
            a=0
            b=0
            filtr_data=[]
            filtr_data= extract_filter_data(data,attrb_idx,attrib)
            m,data_extrc = extract_uniquevalue(filtr_data,class_indx)            
            a=float(data_extrc.count(class_var))            
            b=float(len(data_extrc))            
            fraction = float(a)/float(b+eps)            
            entropy += -fraction*np.log2(fraction+eps)
            
        fraction2 = b/len(data)
        entrpy_1 += -fraction2*entropy
    
    return abs(entrpy_1)

#Filter and extract the array from given array based in condition
def extract_filter_data(data,col_idx,col_val):

    out_data=[]
    for i in range(0,len(data)):
        row =data[i]
        if(row[col_idx] == col_val):
            out_data.append(row) 
    return out_data

def apply_majority_class(data):
    
    un, dt =extract_uniquevalue(data,class_indx)
    cn=[]
    majclass=''
    for value in un: 
        cn.append(dt.count(value))
    if(len(cn)>0):
        if(len(cn)==1):
            majclass=un[0]
        if((len(cn)==2)):
            if(cn[0]>cn[1]):
                majclass=un[0]
            else:
                majclass=un[1]
    return majclass  
    
# Apply greedy policy to find pure Node for tree
def greedy_search_node(data):
    Entrpy_att = []
    InfoGain = []
    for k in range(0,len(col_names)-1):
        gn=find_entropy_base(data)-find_entropy_attrb(data,k)
        InfoGain.append(gn)
    
    if(len(node_cntr) > 0):
        for m in range(0,len(node_cntr)):
            n = node_cntr[m]
            InfoGain[n]=0  # implemented node should not be repeated
      
    node = col_names[np.argmax(InfoGain)]
    node_idx = np.argmax(InfoGain)
    node_cntr.append(node_idx)
    
    return node,node_idx


#Build a Tree
def buildTree(data,depth,tree=None): 
    
    #Apply this on abrupt tree termination
    if(depth ==0):
        return apply_majority_class(data)
    
    class_name = col_names[class_indx]       
        
    #apply greedy policy
    node ,node_indx = greedy_search_node(data)#Get column with highest info gain
    
       
    #Get distinct data on node attribute
    atrValue,atrData = extract_uniquevalue(data,node_indx)
    
    #Create an empty dictionary to create tree    
    if tree is None:        
        tree={}
        tree[node] = {}
          
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in atrValue:        
        sub_table = extract_filter_data(data,node_indx,value)        
        clVal,clData = extract_uniquevalue(sub_table,class_indx) 
        if (len(clVal)==1):
            tree[node][value] = clVal[0]        
        else:        
            tree[node][value] = buildTree(sub_table,depth-1) #Calling the function recursively
                            
    return tree


# This code will predict data based on Tree created
def predict_using_tree(inst,tree):    
    
    for nodes in tree.keys():        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
          
        if type(tree) is dict:
            prediction = predict_using_tree(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction


#This code will initiate testing and , write output to file
def run_test(data,file,oplist):
    zro_cls= apply_majority_class(datasets)
    no_of_test_data=len(data)
    if(no_of_test_data>0):    
        for c in data:     
            data=c
            if(max_depth==0):
                label =zro_cls
                oplist.append(label)
                file.writelines(str(label)+'\n')
            else:
                label =predict_using_tree(data,tree)            
                oplist.append(label)
                file.writelines(str(label)+'\n')

# calculate the error rate for classification
def calc_error(real_data,res):
    
    total =len(real_data)
    cntr=0
    unq, orgnl =extract_uniquevalue(real_data,class_indx)# extract class column from data
    
    for i in range (0,total):
        if(orgnl[i]==res[i]):
            cntr= cntr+1
    error = float(total - cntr)/float(total+eps)
    print(error)
    return error
        
#print (json.dumps(tree, indent=3))
def print_tree(d):
    def pretty(d, indent):
        for i, (key, value) in enumerate(d.items()):
            if isinstance(value, dict):
                node_val =next(iter(value))
                print ('{0}"{1}":'.format( '|' * indent, str(key)))
                pretty(value, indent+1)
                if i == len(d)-1:
                     print ('{0}'.format( '|' * indent))
                else:
                     print ('{0}'.format( '|' * indent))
            else:
                if i == len(d)-1:
                    print ('{0}"{1}": "{2}"'.format( '|' * indent, str(key), value))
                else:
                    print ('{0}"{1}": "{2}",'.format( '|' * indent, str(key), value))
    
    pretty(d,indent=0)    

# Start execution of all the codes here...................................
tree={}
mx_d =max_depth
if(mx_d==0):
    tree={}
else:
    tree=buildTree(datasets,mx_d)
import pprint
#pprint.pprint(tree)

# create test data to be tested
fileTrainO = open(path2,"w")
fileTestO = open(path3,"w")
fileM = open(path4,"w")

# apply decesion tree
run_test(copy_train,fileTrainO,outpt_h_tr)   # test on test data
run_test(test,fileTestO,outpt_h_te)    # test on training data

errtrn =calc_error(datasets,outpt_h_tr)
fileM.writelines("error(train): "+str(errtrn)+'\n')
errtst=calc_error(t_dat,outpt_h_te)
fileM.writelines("error(test): "+str(errtst)+'\n')
#use test data for prediction


fileTrainO.close()
fileTestO.close()
fileM.close()

#print (json.dumps(tree, indent=3))
print_tree(tree)

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
