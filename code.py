import sys
import numpy as np
import statistics
import math
import random
import csv

#const
modelnum = 3 #jumlah model
datatraineachmodel = 298 #jumlah data train setiap model

#functions
def fixText(text): #read csv
    row = []
    z = text.find(',')
    if z == 0:  row.append('')
    else:   row.append(text[:z])
    for x in range(len(text)):
        if text[x] != ',':  pass
        else:
            if x == (len(text)-1):  row.append('')
            else:
                if ',' in text[(x+1):]:
                    y = text.find(',', (x+1))
                    c = text[(x+1):y]
                else:   c = text[(x+1):]
                row.append(c)
    return row

def createTuple(oldFile): #read csv
    ## oldFile is filename (e.g. 'sheet.csv')
    f1 = open(oldFile, "r")
    next(f1) #skip first line
    tup = []
    while 1:
        text = f1.readline()
        if text == "":  break
        else:   pass
        if text[-1] == '\n':
            text = text[:-1]
        else:   pass
        row = fixText(text)
        tup.append(row)
    return tup

def getMeanandStdev(data):
	x11 = [] #x atribut 1 kelas 1
	x12 = [] #x atribut 1 kelas 2
	x21 = [] #x atribut 2 kelas 1
	x22 = [] #x atribut 2 kelas 2
	jumlah1 = 0
	jumlah2 = 0
	for i in data:
		if(i[2] == 1):
			x11.append(i[0])
			x21.append(i[1])
			jumlah1 += 1
		elif(i[2] == 2):
			x12.append(i[0])
			x22.append(i[1])
			jumlah2 += 1
	res11 = {'mean' : statistics.mean(x11), 'stdev': statistics.stdev(x11), 'jumlah': jumlah1}
	res12 = {'mean' : statistics.mean(x12), 'stdev': statistics.stdev(x12), 'jumlah': jumlah2}
	res21 = {'mean' : statistics.mean(x21), 'stdev': statistics.stdev(x21), 'jumlah': jumlah1}
	res22 = {'mean' : statistics.mean(x22), 'stdev': statistics.stdev(x22), 'jumlah': jumlah2}
	out = []
	out.append(res11)
	out.append(res12)
	out.append(res21)
	out.append(res22)
	return out
	
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom
	
def naivebayes(data, meanstdev):
	one = normpdf(data[0], meanstdev[0]['mean'], meanstdev[0]['stdev']) * normpdf(data[1], meanstdev[2]['mean'], meanstdev[2]['stdev']) * (meanstdev[0]['jumlah']/datatraineachmodel)
	two = normpdf(data[0], meanstdev[1]['mean'], meanstdev[1]['stdev']) * normpdf(data[1], meanstdev[3]['mean'], meanstdev[3]['stdev']) * (meanstdev[1]['jumlah']/datatraineachmodel)
	if(one > two):
		return -1 #class 1
	else:
		return 1 #class 2

data1 = np.array(createTuple('TrainsetTugas4ML.csv'))
dataTrain = data1.astype(np.float) #change type to float
dataTrainnumrows = len(dataTrain) - 1
data2 = createTuple('TestsetTugas4ML.csv')
for i in data2:
	i.pop() #removes last column
data3 = np.array(data2)
dataTest = data3.astype(np.float) #change type to float
#print(dataTrain[0])
#meanthing = getMeanandStdev(dataTrain)
#print(naivebayes(dataTest[0],meanthing))

#make the models
model = []
for i in range(modelnum):
	new = []
	for j in range(datatraineachmodel):
		randomindex = random.randint(0,dataTrainnumrows)
		new.append(dataTrain[randomindex])
	model.append(new)
#print(model[0][0][2])

#finding the mean and stdev for each model
meanstdev = []
for i in range(modelnum):
	meanstdev.append(getMeanandStdev(model[i]))

#testing
output = []
for i in dataTest:
	sum = 0
	for j in meanstdev:
		sum += naivebayes(i,j)
	if(sum > 0): #jika sum positif, maka class 2
		output.append(2)
	else: #jika sum negatif, maka class 1
		output.append(1)
output = np.array(output)

# Write CSV 
with open('TebakanTugas4ML.csv', "w", newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for row in range(0,output.shape[0]):
		myList = []
		myList.append(output[row])
		writer.writerow(myList)