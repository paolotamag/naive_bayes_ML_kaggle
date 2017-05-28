import pandas as pd
import numpy as np
import time

#1536242 - Paolo Tamagnini - Project 3 - FDS - Sapienza - 02/2016

print 'Acquiring data..'

#the first part won't have any comment cause it will be just
#the same of forSub.py

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

ncrimeTrain = len(dfTrain)
ncrtimeTest = len(dfTest)

categ = sorted(set(dfTrain['Category']))
nc = len(categ)
dizCat = dict(zip(categ,range(0,nc)))

def diz_n(s):
	setTrain = set(dfTrain[s])
	setTest = set(dfTest[s])
	inTestNotInTrain = setTest - setTrain
	mergedList = list(setTrain) + list(inTestNotInTrain)
	mergedList = sorted(mergedList)
	nMerged = len(mergedList)
	dizTot = dict(zip(mergedList,range(0,nMerged)))
	return dizTot

	

dizQ = diz_n('PdDistrict')
nq = len(dizQ)

dizW = diz_n('DayOfWeek')
nw = len(dizW)

dizS = diz_n('Address')
ns = len(dizS)

def createTimeColumn(df):
	year = []
	month = []
	day = []
	hour = []
	monthDay = []
	yearMonthDay =[]
	yearMonth = []
	hourMinute = []
	
	for i in range(0,len(df)):
		date = df['Dates'][i].replace(':',' ')
		date = date.replace('-',' ')
		date = date.split(' ')
		monthDay.append(date[1]+'-'+date[2])
		yearMonth.append(date[0]+'-'+date[1])
		yearMonthDay.append(date[0]+'-'+date[1]+'-'+date[2]) 
		year.append(date[0])
		month.append(date[1])
		day.append(date[2])
		hour.append(date[3])
		hourMinute.append(date[3]+'-'+date[4])

	df['Year'] = year
	df['Month'] = month
	df['Day'] = day
	df['Hour'] = hour
	df['monthDay'] = monthDay
	df['yearMonth'] = yearMonth
	df['yearMonthDay'] = yearMonthDay    
	df['hourMinute'] = hourMinute

createTimeColumn(dfTrain)
createTimeColumn(dfTest)

dizD = diz_n('Day')
nd = len(dizD)

dizM = diz_n('Month')
nm = len(dizM)

dizY = diz_n('Year')
ny = len(dizY)

dizH = diz_n('Hour')
nh = len(dizH)

dizhm = diz_n('hourMinute')
nhm = len(dizhm)

dizym = diz_n('yearMonth')
nym = len(dizym)

dizmd = diz_n('monthDay')
nmd = len(dizmd)

print 'Dividing the train.csv in 2 dataframe'
print 'by checking if a row has spare or even index..'

#HERE IS EVERYTHING ABOUT THIS FILE
#I DIVIDE THE TRAIN IN 2 DATASET
#ONE WILL BECOME MY NEW TRAIN (dfTrainEven)
#THE OTHER MY NEW TEST (dfTrainSpare).
#MY NEW TEST WILL HAVE CATEGORIES FOR EACH CRIME. 
#THAT MEANS I'LL BE ABLE TO SAY
#IF I AM OVERFITTING THE TRAIN OR NOT

#BY INCREASING THE NUMBER OF TINY RECTANGLES (nl) FOR EXAMPLE
#YOU SEE A DECREASE OF THE LOGLOSS IF YOU USE THE ENTIRE TRAIN TO BUILD THE MATRIX
#AND AGAIN YOU USE IT TO CALCULATE THE PROBABILITY ESTIMATES AND THE LOGLOSS

#INSTEAD BY BUILDING THE MATRIXES ONLY ON THE ROWS WITH AN EVEN INDEX
#AND CALCULATING THE ESTIMATES AND THE RELATIVE LOGLOSS ONLY ON THE ROWS
#WITH A SPARE INDEX YOU ARE ABLE TO SEE THAT THIS DOESN'T HAPPEN ANYMORE

#BECAUSE OF THE LIMITED TRIES ON KAGGLE I WAS ABLE TO SEE BETTER
#WETHER A NEW COMBINATION OF FEATURES WITH A DIFFERENT k AND A DIFFERENT nl
#WAS BETTER OR NOT OF WHAT I HAS SO FAR WITHOUT THE FEAR OF OVERFITTING
dfTrainEven = dfTrain[0::2]
dfTrainSpare = dfTrain[1::2]
dfTrainEven.index = range(0,len(dfTrainEven))
dfTrainSpare.index = range(0,len(dfTrainSpare))

dfMapTrain = dfTrain.loc[dfTrain['X']!=-120.5]
dfMapTest = dfTest.loc[dfTest['X']!=-120.5]

minYTrain = min(dfMapTrain['Y'])
minYTest = min(dfMapTest['Y'])
minYdata = min([minYTrain,minYTest])

#A rectangle has be drawn over S.F. using
#https://developers.google.com/maps/documentation/javascript/examples/rectangle-event 
#New north-east corner: 37.8130801107315, -122.34870163726805
#New south-west corner: 37.68210508088171, -122.51727945709217

maxX = -122.34870163726805
minX = -122.51727945709217
maxY = 37.8130801107315
minY = minYdata
hugebase = maxX - minX
hugeheight = maxY - minY
#THIS CAN BE CHANGED TO TEST DIFFERENT LOGLOSS
#OF COURSE IT WILL BE RELEVANT ONLY IF YOU DECIDE
#TO USE fattL
nl = 154**2
lx = ((hugebase/np.sqrt(nl)))
ly = ((hugeheight/np.sqrt(nl)))

dL = {}
i = 0
for t in range(0,int(np.sqrt(nl))):
    for k in range(0,int(np.sqrt(nl))):
        dL[(t,k)] = i
        i = i + 1
		
start = time.time()
L = np.zeros((nl,nc))

Y = np.zeros((ny,nc))
M = np.zeros((nm,nc))
D = np.zeros((nd,nc))
H = np.zeros((nh,nc))
S = np.zeros((ns,nc))
Q = np.zeros((nq,nc))
W = np.zeros((nw,nc))
HM = np.zeros((nhm,nc))
YM = np.zeros((nym,nc))
MD = np.zeros((nmd,nc))

print 'Building the matrixes from a part of the train.csv..'
#CALCULATING MATRIXES OVER THE ROWS WITH EVEN INDEX
for i in range(0,len(dfTrainEven)):
	
	if i%100000 == 0 and i!=0:
		print i, '/',len(dfTrainEven),'crimes have been processed.'
		
	if(dfTrainEven['X'][i]>=minX and dfTrainEven['X'][i]<=maxX and dfTrainEven['Y'][i]<=maxY and dfTrainEven['Y'][i]>=minY):   
		t = int(np.floor((dfTrainEven['X'][i]-minX)/lx))
		k = int(np.floor((dfTrainEven['Y'][i]-minY)/ly))
		L[dL[(t,k)],dizCat[dfTrainEven['Category'][i]]] += 1
	
	HM[dizhm[dfTrainEven['hourMinute'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	Q[dizQ[dfTrainEven['PdDistrict'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	S[dizS[dfTrainEven['Address'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	M[dizM[dfTrainEven['Month'][i]],dizCat[dfTrainEven['Category'][i]]] += 1 
	D[dizD[dfTrainEven['Day'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	H[dizH[dfTrainEven['Hour'][i]],dizCat[dfTrainEven['Category'][i]]] += 1  
	W[dizW[dfTrainEven['DayOfWeek'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	Y[dizY[dfTrainEven['Year'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	YM[dizym[dfTrainEven['yearMonth'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	MD[dizmd[dfTrainEven['monthDay'][i]],dizCat[dfTrainEven['Category'][i]]] += 1
	
end = time.time()
print 'Time for calculating matrixes:'
print int(end - start)/60,'m',int((end - start)%60),'s'

probCateg = np.zeros(nc)
freqCat = np.zeros(nc)
for i in range(0,nc):
    f = sum(HM[:,i])
    freqCat[i]=f
    r = f /len(dfTrainEven)
    probCateg[i]=r
	
def formulaMat(A,k):
	
	nr = len(A)
	nc = len(A[0])
	B = np.zeros((nr,nc))
	for i in range(0,nr):
		for j in range(0,nc):
			B[i,j] = (A[i,j]+k)/(freqCat[j]+k*nr)
	return B

#THIS CAN BE CHANGED TO TEST DIFFERENT LOGLOSS
beta = 0.95
Ynew = formulaMat(Y,beta)
YMnew = formulaMat(YM,beta)
Mnew = formulaMat(M,beta)
Dnew = formulaMat(D,beta)
Hnew = formulaMat(H,beta)
Qnew = formulaMat(Q,beta)
Wnew = formulaMat(W,beta)
Snew = formulaMat(S,beta)
Lnew = formulaMat(L,beta)
HMnew = formulaMat(HM,beta)
MDnew = formulaMat(MD,beta)

fattLexcep = np.zeros(nc)
for i in range(0,nc):
	fattLexcep[i]=sum(Lnew[:,i])/nl
	
print 'Calculating the probabilities estimates' 
print 'from the other part of the train.csv..'
subTrain = []
start = time.time()
#CALCULATING PROBABILITIES ESTIMATES OVER ROWS WITH SPARE INDEX
for i in range(0,len(dfTrainSpare)):

	if i %100000 == 0 and i!=0:
		print i, '/',len(dfTrainSpare),'crimes have been processed.'
		
	if(dfTrainSpare['X'][i]>=minX and dfTrainSpare['X'][i]<=maxX and dfTrainSpare['Y'][i]<=maxY and dfTrainSpare['Y'][i]>=minY):   
	
		t = int(np.floor((dfTrainSpare['X'][i]-minX)/lx))
		k = int(np.floor((dfTrainSpare['Y'][i]-minY)/ly))
		fattL=Lnew[dL[(t,k)],:]
		
	else:
	
		fattL=fattLexcep
		
		
	fattHM = HMnew[dizhm[dfTrainSpare['hourMinute'][i]],:]
	fattQ = Qnew[dizQ[dfTrainSpare['PdDistrict'][i]],:]
	fattH = Hnew[dizH[dfTrainSpare['Hour'][i]],:]
	fattW = Wnew[dizW[dfTrainSpare['DayOfWeek'][i]],:]
	fattS = Snew[dizS[dfTrainSpare['Address'][i]],:]
	fattY = Ynew[dizY[dfTrainSpare['Year'][i]],:]
	fattM = Mnew[dizM[dfTrainSpare['Month'][i]],:]
	fattD = Dnew[dizD[dfTrainSpare['Day'][i]],:]
	fattYM = YMnew[dizym[dfTrainSpare['yearMonth'][i]],:]
	fattMD = MDnew[dizmd[dfTrainSpare['monthDay'][i]],:]
	
	#MAKE SURE YOU CHOOSE THOSE CORRECTLY
	listiz = list(fattL*fattW*fattY*fattHM*probCateg)
	
	subTrain.append(listiz)

end = time.time()
print 'Time for calculating estimates for each crime:'
print int(end - start)/60,'m',int((end - start)%60),'s'

	
categ = list(dfTrainSpare['Category'])
categ = sorted(list(set(categ)))
dizCatReverse = dict(zip(range(0,nc),categ))
subDFTrain = pd.DataFrame(subTrain,columns=categ)

#CALCULATING LOGLOSS FOR subDFTrain
#SEE https://www.kaggle.com/c/sf-crime/details/evaluation
yij = np.zeros((len(dfTrainSpare),nc))
for i in range(0,len(dfTrainSpare)):
	yij[i,dizCat[dfTrainSpare['Category'][i]]] = 1

sumSub = subDFTrain.sum(axis=1)
summa = []
for i in range(0,len(dfTrainSpare)):
	for j in range(0,nc):
		if yij[i,j] == 1.0:
			one = min(subDFTrain[dizCatReverse[j]][i],1-10**(-15))
			two = max(one,10**(-15))
			three = two/sumSub[i]
			ad = np.log(three)
			summa.append(ad)
			
#RESULT WILL BE ACTUALLY THE LOGLOSS VALUE OF OUR ESTIMATES
#IT NORMALLY GIVES AROUND 2.36 FOR CHOICES OF FEATURES THAT ON KAGGLE LEATHERBOARD
#GIVE 2.34 ... NOT TOO BAD!
result = (-1/float(len(dfTrainSpare)))*sum(summa)

#JUST FOR FUN WE ALSO CALCULATE THE % OF SUCCESS.
#THAT MEANS TAKING THE MAXIMUM PROBABILITY FOR EACH ROW
#OF subDFTrain,THEN WE GET THE RELATIVE CATEGORY
#AND WE COUNT WITH c IF THE GUESS IS RIGHT OR NOT.
#IN THE END WE DEVIDE c BY THE NUMBER OF CRIMES WITH SPARE INDEX
#AND SO WE GET OUR PERCENTAGE WHICH IS USUALLY AROUND 30%
maxIDSub = subDFTrain.idxmax(axis=1)
c = 0
for i in range(0,len(dfTrainSpare)):
	if dfTrainSpare['Category'][i] == maxIDSub[i]:
		c = c + 1
percentage =  c / float(len(dfTrainSpare))


print 'The estimated logloss with this combination is:', result
print 'The estimated percentage of categories correctly guessed'
print 'with this combination is:',percentage*100,'%'

