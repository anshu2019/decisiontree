
# coding: utf-8

# In[54]:


# This is a Inspection Class to inspect data and Find entropy at root node
# This class will find error rate using Majority vote class method as well

# Import the required librray
import sys
import numpy as np
import re
import csv
eps = np.finfo(float).eps
from numpy import log2 as log

path = sys.argv[1]
out_file =sys.argv[2]
documents =[]
class_labels =[]
unique_labels =[]
cls_indx=0
entropy =0
error_rate=0

#read the file
print(path)
with open(path, 'r') as input_csv:            
            reader = csv.reader(input_csv) 
            next(reader, None)  # skip the headers
            documents = [r for r in reader]
            
if(len(documents) > 0):
    cls_indx =len(documents[0])-1 # class label exist at tis index

    
#extract the unique class label
for i in range (0,len(documents)):
    label = documents[i][cls_indx]
    class_labels.append(label)
    # add unique label
    if label not in unique_labels: 
            unique_labels.append(label)   
    
#print(class_labels)
#print(unique_labels)

# calculate entropy
x =class_labels.count(unique_labels[0])
y = class_labels.count(unique_labels[1])
t = x+y

for u in unique_labels:
    fraction= 0
    fraction = float(class_labels.count(u))/  float(t +eps)
    entropy += -fraction*np.log2(fraction + eps)

print ("entropy calculated : %f" %(abs(entropy)))

# calculate error rate ,using Majority class vote

if(x>y):
    error_rate = float(y)/float(t)
else:
    error_rate = float(x)/float(t)

print("entropy calculated : %f" %(error_rate))

entrpy = "Entropy : " + str(abs(entropy)) 
err = " And Error_Rate: " + str(error_rate)
#write to output files
fileo = open(out_file,"w")
fileo.writelines("entropy: " + str(abs(entropy)) +'\n')
fileo.writelines("error: " + str((error_rate)) +'\n')

