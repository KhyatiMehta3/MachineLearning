#Simple Linear Regression model.

#Getting data from files - Day3. File in progress.

f = open('height_weight_lineaRegression_data.txt','r')
ht = [] # initialise height and weight lists
wt = [] 
lines = f.readlines()[1:] # To skip header line
#Append heights and weights from the files
for line in lines:
    ht.append(line.split('\t')[0])
    wt.append(line.split('\t')[1].strip('\n'))
