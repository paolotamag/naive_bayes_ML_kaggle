import pandas as pd
import numpy as np
import time

#1536242 - Paolo Tamagnini - Project 3 - FDS - Sapienza - 02/2016

print 'Acquiring data..'
#CREATING PANDAS DFS FROM THE CSVs
dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

ncrimeTrain = len(dfTrain)
ncrtimeTest = len(dfTest)

#BUILDING A DICTIONARY THAT FOR EACH KIND OF CRIME RETURNS 
#A DIFFERENT NUMBER BETWEEN 0 AND (#DIFFERENT KIND OF CRIMES) - 1
categ = sorted(set(dfTrain['Category']))
nc = len(categ)
dizCat = dict(zip(categ,range(0,nc)))

#DEFINING A FUNCTION ABLE TO CHECK ALL POSSIBLE VALUES
#FOR COLUMNS THAT ARE PRESENT BOTH IN TRAIN AND IN TEST.
#THEN WITH A COMPLETE SET IT CREATES A FULL DICTIONARY 
#THAT FOR EACH POSSIBLE VALUE OF THE COLUMN s IT RETURNS 
#A DIFFERENT NUMBER BETWEEN 0 AND THE 
#(#DIFFERENT VALUES IN COLUMNS s OF TRAIN AND TEST) - 1
def diz_n(s):
	setTrain = set(dfTrain[s])
	setTest = set(dfTest[s])
	inTestNotInTrain = setTest - setTrain
	mergedList = list(setTrain) + list(inTestNotInTrain)
	mergedList = sorted(mergedList)
	nMerged = len(mergedList)
	dizTot = dict(zip(mergedList,range(0,nMerged)))
	return dizTot

	
#CREATING THE DICTIONARY FOR PdDistrict
dizQ = diz_n('PdDistrict')
nq = len(dizQ)

#CREATING THE DICTIONARY FOR DayOfWeek
dizW = diz_n('DayOfWeek')
nw = len(dizW)

#CREATING THE DICTIONARY FOR Address
dizS = diz_n('Address')
ns = len(dizS)

#IN ORDER TO GET A DICTIONARY AND THE INFO REGARDING TIME
#WE WILL SPLIT THE DateTimeIndex IN MANY COLUMNS THAT WILL BE ADDED
#TO BOTH DATAFRAMES SO WE CAN PICK YEAR, MONTH, DAY, HOUR AND THEIR COMBINATION
#SEPARATELY. TO DO THIS WE DEFINE A FUNCTION 
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
		hourMinute.append(date[3]+':'+date[4])

	df['Year'] = year
	df['Month'] = month
	df['Day'] = day
	df['Hour'] = hour
	df['monthDay'] = monthDay
	df['yearMonth'] = yearMonth
	df['yearMonthDay'] = yearMonthDay    
	df['hourMinute'] = hourMinute

#WE APPLY THE FUNCTION TO BOTH TEST AND TRAIN
createTimeColumn(dfTrain)
createTimeColumn(dfTest)

#NOW WE CAN CREATE OUR DICTIONARY FOR THE NEW COLUMNS

#CREATING THE DICTIONARY FOR Day
dizD = diz_n('Day')
nd = len(dizD)

#CREATING THE DICTIONARY FOR Month
dizM = diz_n('Month')
nm = len(dizM)

#CREATING THE DICTIONARY FOR Year
dizY = diz_n('Year')
ny = len(dizY)

#CREATING THE DICTIONARY FOR Hour
dizH = diz_n('Hour')
nh = len(dizH)

#CREATING THE DICTIONARY FOR hourMinute
dizhm = diz_n('hourMinute')
nhm = len(dizhm)

#CREATING THE DICTIONARY FOR yearMonth
dizym = diz_n('yearMonth')
nym = len(dizym)

#CREATING THE DICTIONARY FOR monthDay
dizmd = diz_n('monthDay')
nmd = len(dizmd)


#WE WILL NOW BUILD A MODEL TO GET INFO FROM THE X AND THE Y COLUMN AS WELL.
#TO DO SO WE MUST FIRST TAKE OUT EVERY CRIME WITH UNKNOWN COORDINATES.
#THOSE CRIMES WILL RESULT ALL WITH A DEFAULT VALUE OF
#X = -120.5
#Y = 90.0
#TAKING AWAY EVERY CRIM WITH X = -120.5 WILL WORK
dfMapTrain = dfTrain.loc[dfTrain['X']!=-120.5]
dfMapTest = dfTest.loc[dfTest['X']!=-120.5]

#WE WANT TO DRAW NOW A RECTANGLE AROUND SAN FRANCISCO
#WE WILL DIVIDE IT MANY RECTANGLES OF SAME DIMENSION
#AND WITH THE SAME RATIO OF THE BIG ONE

#SAN FRANCISCO IS SURROUNDED BY WATER EXCEPT FROM THE SOUTH
#FOR THIS REASON WE BOUND ONLY THE BOTTOM OF THE BIG RECTANGLE
#WITH THE LOWEST Y WE CAN FIND BETWEEN X AND Y
minYTrain = min(dfMapTrain['Y'])
minYTest = min(dfMapTest['Y'])
minYdata = min([minYTrain,minYTest])

#BECAUSE OF SOME CRIMES HAPPENING FAR AWAY FROM THE CITY (IN THE MIDDLE OF THE SEA)
#WE ARE DRAWING ALONG THE COAST THE OTHER 3 SIDES MANUALLY USING
#https://developers.google.com/maps/documentation/javascript/examples/rectangle-event 
#north-east corner: 37.8130801107315, -122.34870163726805
#south-west corner: 37.68210508088171, -122.51727945709217

#WE KNOW NOW THE BOUNDS THAT DETERMINE OUR RECTANGLE
maxX = -122.34870163726805
minX = -122.51727945709217
maxY = 37.8130801107315
minY = minYdata
#THE WIDTH OF THE RECTANGLE
hugebase = maxX - minX
#THE HEIGHT OF THE RECTANGLE
hugeheight = maxY - minY
#THE NUMBER OF TINY RECTANGLES WE ARE DIVIDING IN SF
nl = 154**2
#THE WIDTH OF THE TINY RECTANGLE
lx = ((hugebase/np.sqrt(nl)))
#THE HEIGHT OF THE TINY RECTANGLE
ly = ((hugeheight/np.sqrt(nl)))

#TO KNOW IN WICH TINY RECTANGLE A CRIME FELL WITH HIS COORDINATES
#WE WILL USE A DICTIONARY THAT GIVEN A TUPLE OF 2 INTEGERS t,k
#IT GIVES BACK A NUMBER THAT IT WILL BE BOUNDED TO ONLY ONE OF 
#THE TINY RECTANGLE. t AND k ARE CALCULATED FROM THE COORDINATES
#OF THE CRIMES (SEE THE PROJECT DESCRIPTION)
dL = {}
i = 0
for t in range(0,int(np.sqrt(nl))):
    for k in range(0,int(np.sqrt(nl))):
        dL[(t,k)] = i
        i = i + 1

#WE CREATE AN EMPTY MATRIX FOR EACH FEATURE
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

print 'Building the matrixes from train.csv..'

#WE WILL NOW GO THROUGH EVERY CRIME IN THE TRAIN
for i in range(0,ncrimeTrain):
    #THIS IS TO VISUALIZE THE PROCESS
	if i%100000 == 0 and i!=0:
		print i, '/',ncrimeTrain,'crimes have been processed.'
	#IF THE COORDINATE ARE WITHIN THE BIG SQUARE ON SAN FRANCISCO	
	if(dfTrain['X'][i]>=minX and dfTrain['X'][i]<=maxX and dfTrain['Y'][i]<=maxY and dfTrain['Y'][i]>=minY):
		#WE COMPUTE THE NUMBER OF TINY RECTANGLES BETWEEN OUR X AND minX
		t = int(np.floor((dfTrain['X'][i]-minX)/lx))
		#WE COMPUTE THE NUMBER OF TINY RECTANGLES BETWEEN OUR Y AND minY
		k = int(np.floor((dfTrain['Y'][i]-minY)/ly))
		#WE FIND THANKS TO t AND k THE ROW OF THE MATRIX L CONNECTED WITH dL 
		#TO THE TINY RECTANGLE THE CRIME IS, AND WE FIND WITH dizCat THE COLUMN OF
		#THE KIND OF THE CURRENT CRIME, THEN WE HAD +1 TO THAT ELEMENT OF THE MATRIX
		#TO INCREASE THE NUMBER OF CRIMES HAPPENED IN THAT TINY RECTANGLE
		L[dL[(t,k)],dizCat[dfTrain['Category'][i]]] += 1
	#THANKS TO ALL THE DICTIONARIES WE HAVE BUILT, WE DO THE SAME THING FOR EACH FEATURE (ex: DayOfWeek).
	#GIVEN THE ROW CONNECTED TO THE ACUAL VALUE OF THE FEATURE OF THE CRIME (ex: Monday),
	#AND THE VALUE OF THE KIND OF CRIME (ex: Theft), WE INCREASE THE ELEMENT OF THE MATRIX BY 1
	#(ex: W(Monday,Theft) += 1) TO SAY THAT THERE IS ONE MORE CRIM OF THAT KIND WITH THAT FEATURE
	HM[dizhm[dfTrain['hourMinute'][i]],dizCat[dfTrain['Category'][i]]] += 1
	Q[dizQ[dfTrain['PdDistrict'][i]],dizCat[dfTrain['Category'][i]]] += 1
	S[dizS[dfTrain['Address'][i]],dizCat[dfTrain['Category'][i]]] += 1
	M[dizM[dfTrain['Month'][i]],dizCat[dfTrain['Category'][i]]] += 1 
	D[dizD[dfTrain['Day'][i]],dizCat[dfTrain['Category'][i]]] += 1
	H[dizH[dfTrain['Hour'][i]],dizCat[dfTrain['Category'][i]]] += 1  
	W[dizW[dfTrain['DayOfWeek'][i]],dizCat[dfTrain['Category'][i]]] += 1
	Y[dizY[dfTrain['Year'][i]],dizCat[dfTrain['Category'][i]]] += 1
	YM[dizym[dfTrain['yearMonth'][i]],dizCat[dfTrain['Category'][i]]] += 1
	MD[dizmd[dfTrain['monthDay'][i]],dizCat[dfTrain['Category'][i]]] += 1

end = time.time()

print 'Time for calculating matrixes:'
print int(end - start)/60,'m',int((end - start)%60),'s'

#WE NEED TO COMPUTE A VECTOR THAT FOR EACH CATEGORIES 
#IT GIVES THE % OF CRIMES OF THAT KIND OVER ALL THE ONES IN THE TRAIN.

#TO DO THIS WE USE ONE OF THE MATRIX (we picked HM)
#AND WE SUM A COLUMN TO KNOW THE OVERALL NUMBER OF CRIMES
#OF THE RELATIVE KIND, SAVING IN freqCat THOSE VALUES FOR LATER,
#THEN WE JUST DEVIDE EACH OF THOSE VALUE FOR THE NUMBER OF CRIME
#IN THE TRAIN (ncrimeTrain) OBTAINING THE VECTOR probCateg

probCateg = np.zeros(nc)
freqCat = np.zeros(nc)
for i in range(0,nc):
    f = sum(HM[:,i])
    freqCat[i]=f
    r = f /ncrimeTrain
    probCateg[i]=r

	
#TO APPLY THE FORMULA IN THE DESCRIPTION OF THE PROJECT
#WE DEFINE A FUNCTION ABLE TO RETURN A MATRIX OF PROBABILITIES
#GIVEN A MATRIX OF FREQUENCIES AND THE 
#FREQUENCIES OF KIND OF CRIMES (freqCat)
def formulaMat(A,k):
    
    nr = len(A)
    nc = len(A[0])
    B = np.zeros((nr,nc))
    for i in range(0,nr):
        for j in range(0,nc):
            B[i,j] = (A[i,j]+k)/(freqCat[j]+k*nr)
    return B
	
#CREATING NEW MATRIXES WITH SAME PARAMETER beta
#ALSO CHOOSING beta WILL INFLUENCE THE RESULT ON KAGGLE
#I USE toCheckSub.py TO PREDICT WICH beta IS BETTER
#WITH A CERTAIN COMBINATION
beta = 0.9
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

#THIS VECTOR OF PROBABILITIES IS GIVEN BY
#COMPUTING THE AVERAGE OF EACH COLUMN OF L
#AND IT WILL BE USED FOR EACH CRIME WITH UNKOWN
#COORDINATES OR THAT DOESN'T FALL IN OUR BIG RECTANGLE
#COVERING MOST OF SAN FRANCISCO
fattLexcep = np.zeros(nc)
for i in range(0,nc):
    fattLexcep[i]=sum(Lnew[:,i])/nl

#NOW WE WILL COMPUTE THE OUTPUT THAT WILL GO IN THE CSV OF THE SUBMISSION
print 'Calculating the probabilities estimates' 
print 'for each crime of test.csv..'

sub = []
start = time.time()
#FOR EACH CRIME IN THE TEST
for i in range(0,ncrtimeTest):

	if i %100000 == 0 and i!=0:
		print i, '/',ncrtimeTest,'crimes have been processed.'
	#IF A CRIME FALLS IN THE BIG RECTANGLE OVER SAN FRANCISCO	
	if(dfTest['X'][i]>=minX and dfTest['X'][i]<=maxX and dfTest['Y'][i]<=maxY and dfTest['Y'][i]>=minY):
		#WE CAN COMPUTE WHICH ROW OF L GIVES THE PROBABILITIES OF THE FEATURE LOCATION
		#THANKS TO t AND k
		t = int(np.floor((dfTest['X'][i]-minX)/lx))
		k = int(np.floor((dfTest['Y'][i]-minY)/ly))
		fattL=Lnew[dL[(t,k)],:]
	#IF IT DOESN'T FALL IN THE BIG RECTANGLE	
	else:
		#JUST TAKE THE AVERAGE PROBABILY VECTOR OF THE MATRIX L
		fattL=fattLexcep
		
	#WE DO THE SAME FOR EACH FEATURE BY PICKING THE CORRECT ROW
	#CORRESPONDING TO THE VALUE OF THE FEATURE OF THE CRIME OF THE TEST
	#FROM THE RELATIVE MATRIX OF THAT FEATURE
	fattHM = HMnew[dizhm[dfTest['hourMinute'][i]],:]
	fattQ = Qnew[dizQ[dfTest['PdDistrict'][i]],:]
	fattH = Hnew[dizH[dfTest['Hour'][i]],:]
	fattW = Wnew[dizW[dfTest['DayOfWeek'][i]],:]
	fattS = Snew[dizS[dfTest['Address'][i]],:]
	fattY = Ynew[dizY[dfTest['Year'][i]],:]
	fattM = Mnew[dizM[dfTest['Month'][i]],:]
	fattD = Dnew[dizD[dfTest['Day'][i]],:]
	fattYM = YMnew[dizym[dfTest['yearMonth'][i]],:]
	fattMD = MDnew[dizmd[dfTest['monthDay'][i]],:]
	
	#HERE COMES ALL THE TROUBLES
	
	#WE NEED TO DECIDE WHICH FEATURE CHOOSE CAUSE THIS WILL
	#CHANGE A LOT THE RESULT OF KAGGLE
	#THE BEST COMBINATION OF FEATURES IS GIVEN THANKS TO THE ESTIMATION MADE
	#WITH toCheckSub.py .
	#ANYWAY THIS IS THE COMBINATION USED TO ACHIEVE THE ACTUAL SCORE
	#I HAVE ON THE LEATHERBOARD
	#listiz WILL BE THE RESULT FOR THE GIVEN CRIME OF THE TEST
	#AND IT WILL BE A LIST CONTAINING A PROBABILITY FOR EACH KIND OF CRIME
	#THAT ESTIMATES THE PROBABILITY THAT CRIME BELONGS TO A KIND OF CRIME AND
	#IT IS COMPUTED BY THE PRODUCT BETWEEN EACH CHOSEN FEATURE ROWS AND probCateg
	
	listiz = list(fattS*fattY*fattW*fattHM*probCateg)
	
	#WE ADD ON TOP OF THE LIST THE INDEX OF THE CRIME OF THE TEST
	listiz.insert(0, i)
	#AND WE APPEND IT IN A LIST WITH A LIST OF ESTIMATES FOR EACH CRIME
	sub.append(listiz)

end = time.time()
print 'Time for calculating estimates for each crime:'
print int(end - start)/60,'m',int((end - start)%60),'s'

#WE CREATE THE DATAFRAME FROM THE LIST OF LIST
#AS DESCRIBED IN KAGGLE
columnia = categ
columnia.insert(0, 'Id')
subDF = pd.DataFrame(sub,columns=columnia)

#WE OUTPUT THE CSV OF THE SUBMISSION
subDF.to_csv('1536242-S_W_Y_HM_k09.csv',index = False)