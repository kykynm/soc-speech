
# coding: utf-8

import torch
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import pandas as pd

def upload_clean_data(filename, filename2):
    fr=open(filename,'r');
    refFilename, stuFilename, rawScoreAccWL, pathLenAccWL, scoreAccW, rawScoreAccSL, pathLenAccSL, scoreAccS, recScore = ([] for i in range(9))
    scoreAccP, rawScoreAccW,pathLenAccW, numOfRef, rawScoreAccS,pathLenAccS  = (0 for i in range(6)) 

    for line in fr:
            arr = line.split();
            if (arr[1] == "SENT_END!"):
                refFilename.append(arr[2])
                stuFilename.append(arr[5])
                recScore.append(float(arr[6]))
                rawScoreAccWL.append(rawScoreAccW) 
                pathLenAccWL.append(pathLenAccW)
                scoreAccW.append(rawScoreAccW/pathLenAccW)
                rawScoreAccW = 0
                pathLenAccW = 0
                rawScoreAccSL.append(rawScoreAccS) 
                pathLenAccSL.append(pathLenAccS)
                scoreAccS.append(rawScoreAccS/pathLenAccS)
                rawScoreAccS = 0
                pathLenAccS = 0
                numOfRef+=1
            if(arr[0] == "w" ) and (arr[1] != "SENT_START!") and  (arr[1] != "WORD_PAUSE!") and (arr[1] !="SENT_END!"):                
                rawScoreAccW += float(arr[10])
                pathLenAccW += float(arr[11])

            elif(arr[0] == "w" ):
                rawScoreAccS += float(arr[10])
                pathLenAccS += float(arr[11])

    fl = open(filename2,'r')
    fault, refFault, stuFault, scoreCSV, pronunciation,stress,intonation,fluency, overall, score, rawScore, pathLen, refName, stuName, scoreSilence, rawScoreSilence, pathLenSilence = ([] for i in range(17))  
    x=0
    cont = 0
    for line in fl:
        arr = line.split("|");
        if (arr[7] == "0") and (arr[8] == "0") and (arr[9] == "0") and (arr[17] == "0") and (arr[18] == "0") and (arr[19] == "0") and(arr[20] == "0"):
            for m in range(numOfRef):
                if (refFilename[m] == arr[3]) and (stuFilename[m] == arr[2] + ".wav"):
                    cont = 1
                    score.append(scoreAccW[m])
                    rawScore.append(rawScoreAccWL[m])
                    pathLen.append(pathLenAccWL[m])
                    refName.append(refFilename[m])
                    stuName.append(stuFilename[m])
                    scoreSilence.append(scoreAccS[m])
                    rawScoreSilence.append(rawScoreAccSL[m])
                    pathLenSilence.append(pathLenAccSL[m])
            if(cont == 0):
                print("No match",  arr[3],  arr[2])
            else:
                cont = 0
                pronunciation.append(float(arr[21]))
                stress.append(float(arr[22]))
                intonation.append(float(arr[23]))
                fluency.append(float(arr[24]))
                overall.append((pronunciation[x]+stress[x]+intonation[x]+fluency[x])/4)
                scoreCSV.append(float(arr[6]))
                x+=1
    
    dictData = {"scoreSilence" : scoreSilence, "rawScoreSilence" : rawScoreSilence, "pathLenSilence" : pathLenSilence, "scoreCSV" : scoreCSV,"refFilename" : refName, "stuFilename" : stuName,"score" : score, "rawScore" : rawScore, "pathLen" : pathLen, "pronunciation" : pronunciation, "stress" : stress, "intonation" : intonation, "fluency" : fluency, "overall" : overall}
    return dictData

def upload_all_data(filename, filename2):

    fr=open(filename,'r');
    phoneme, refFilename, stuFilename, rawScoreAccWL, pathLenAccWL, scoreAccW, rawScoreAccSL, pathLenAccSL, scoreAccS, recScore = ([] for i in range(10))
    numOfP, scoreAccP, rawScoreAccW,pathLenAccW, numOfRef, rawScoreAccS,pathLenAccS  = (0 for i in range(7)) 

    for line in fr:
            arr = line.split();
            if (arr[1] == "SENT_END!"):
                refFilename.append(arr[2])
                stuFilename.append(arr[5])
                recScore.append(float(arr[6]))
                rawScoreAccWL.append(rawScoreAccW) 
                pathLenAccWL.append(pathLenAccW)
                scoreAccW.append(rawScoreAccW/pathLenAccW)
                rawScoreAccW = 0
                pathLenAccW = 0
                rawScoreAccSL.append(rawScoreAccS) 
                pathLenAccSL.append(pathLenAccS)
                scoreAccS.append(rawScoreAccS/pathLenAccS)
                rawScoreAccS = 0
                pathLenAccS = 0
                phoneme.append(numOfP)
                numOfP = 0
                numOfRef+=1
            if(arr[0] == "w" ) and (arr[1] != "SENT_START!") and  (arr[1] != "WORD_PAUSE!") and (arr[1] !="SENT_END!"):                
                rawScoreAccW += float(arr[10])
                pathLenAccW += float(arr[11])

            elif(arr[0] == "w" ):
                rawScoreAccS += float(arr[10])
                pathLenAccS += float(arr[11])
            if (arr[0] == "p") and (arr[1]!="sil"):
                numOfP+=1 

    fl = open(filename2,'r')
    NumOfPho, fault, refFault, stuFault, scoreCSV, pronunciation,stress,intonation,fluency, overall, score, rawScore, pathLen, refName, stuName, scoreSilence, rawScoreSilence, pathLenSilence = ([] for i in range(18))  
    x=0
    cont = 0
    for line in fl:
        arr = line.split("|");
        
        for m in range(numOfRef):
            if (refFilename[m] == arr[3]) and (stuFilename[m] == arr[2] + ".wav"):
                cont = 1
                score.append(scoreAccW[m])
                rawScore.append(rawScoreAccWL[m])
                pathLen.append(pathLenAccWL[m])
                refName.append(refFilename[m])
                stuName.append(stuFilename[m])
                scoreSilence.append(scoreAccS[m])
                rawScoreSilence.append(rawScoreAccSL[m])
                pathLenSilence.append(pathLenAccSL[m])
                NumOfPho.append(phoneme[m])
        if(cont == 0):
            print("No match",  arr[3],  arr[2])
        else:
            cont = 0
            pronunciation.append(float(arr[21]))
            stress.append(float(arr[22]))
            intonation.append(float(arr[23]))
            fluency.append(float(arr[24]))
            overall.append((pronunciation[x]+stress[x]+intonation[x]+fluency[x])/4)
            scoreCSV.append(float(arr[6]))
            if (arr[7] == "1") or (arr[8] == "1") or (arr[9] == "1"):
                refFault.append(1)
                if (arr[17] == "1") or (arr[18] == "1") or (arr[19] == "1") or (arr[20] == "1"):
                    stuFault.append(1)
                    fault.append("both")
                else:
                    stuFault.append(0)
                    fault.append("reference")        
            elif (arr[17] == "1") or (arr[18] == "1") or (arr[19] == "1") or (arr[20] == "1"):
                stuFault.append(1)
                refFault.append(0)
                fault.append("student")
            else:
                stuFault.append(0)
                refFault.append(0)
                fault.append("clear")
            x+=1
    
    dictData = {"scoreSilence" : scoreSilence, "rawScoreSilence" : rawScoreSilence, "pathLenSilence" : pathLenSilence, "scoreCSV" : scoreCSV,"refFilename" : refName, "stuFilename" : stuName,"score" : score, "rawScore" : rawScore, "pathLen" : pathLen, "pronunciation" : pronunciation, "stress" : stress, "intonation" : intonation, "fluency" : fluency, "overall" : overall, "stuFault" : stuFault, "refFault" : refFault, "fault": fault, "phoneme": NumOfPho}
    return dictData

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
#     normalizace, -mean/standard deviation    
    def norm(self, x):
        return (x-self.x_mean)/self.x_stddev
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = F.tanh(self.linear1(self.norm(x)))
        y_pred = self.linear2(h_relu)
        return y_pred
    
class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
  
    def norm(self, x):
        return (x-self.x_mean)/self.x_stddev
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = F.tanh(self.linear1(self.norm(x)))
        h_relu = F.tanh(self.linear2(h_relu))
        y_pred = self.linear3(h_relu)
        return y_pred

def train_NN(x, y, model):    
    for t in range(300):
     # Forward pass: Compute predicted y by passing x to the model
        optimizer.zero_grad()       
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.        
        loss.backward()
        optimizer.step()

def test_NN(x_test, y_test, model):
    model.eval()
    test_loss = 0
    correct = 0
    output = model(x_test)
    test_loss = criterion(output, y_test).data[0] # sum up batch loss\
      
    print('Average loss: ', test_loss)


def graph(model, dataDict, dataForModel, save = False):
    sns.set(color_codes=True)
    np.random.seed(sum(map(ord, "distributions")))

    plt.plot(dataDict["score"], dataDict["overall"], 'ro', label="Score")
    plt.axis([0, 1, -0.3, 10.3])
    plt.xlabel("Computer")
    plt.ylabel("Human")
    if (save):
        plt.savefig("HumanComputerScore.png")
    plt.show()
    
    dat = np.asarray(dataForModel)
    dat = np.transpose(dat)
    dat = Variable(torch.FloatTensor(dat))
    dat = model(dat)

    plt.plot(dat.data.numpy(), dataDict["overall"], 'ro', label="Score")
    plt.axis([0, 10, -0.3, 10.3])
    plt.xlabel("Neural Net")
    plt.ylabel("Human")
    if (save):
        plt.savefig("HumanNNScore.png")
    plt.show()


def saveResults(model, dataDict, dataForModel, filename = "results.txt"):
    textFile = open(filename, "w")
    dat = np.asarray(dataForModel)
    dat = np.transpose(dat)
    dat = Variable(torch.FloatTensor(dat))
    dat = model(dat)
    k = dat.data.numpy()
    k.tolist
    textFile.write("refFilename studentFilename pathLen rawScore scoreModel \n")
    for x in range (len(dataDict["refFilename"])):
        a= dataDict["refFilename"][x]+" "+dataDict["stuFilename"][x]+" "+str(dataDict["pathLen"][x])+" "+str(dataDict["rawScore"][x])+" "+str(k[x][0]) + "\n"
        textFile.write(a)    

    textFile.close()


dataDict = upload_all_data("all.match_newNN", "academusonline-export_new.csv")
lis = [dataDict["score"], dataDict["scoreSilence"], dataDict["pathLen"], dataDict["rawScore"], dataDict["stuFault"]]
lis2 = [dataDict["overall"]]
x = np.asarray(lis)
y = np.asarray(lis2)
x = np.transpose(x)
y = np.transpose(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

x_train = Variable(torch.FloatTensor(x_train)) #input 
y_train = Variable(torch.FloatTensor(y_train)) #output
x_test = Variable(torch.FloatTensor(x_test))
y_test = Variable(torch.FloatTensor(y_test))


D_in, H, D_out = 3, 5, 1
model = TwoLayerNet(D_in, H, D_out)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

x_mean = x_train.mean(dim=0)
x_var = x_train.var(dim=0)
model.x_mean = x_mean
model.x_stddev = x_var.pow(0.5)

train_NN(x_train, y_train, model)
test_NN(x_test, y_test, model)

graph(model, dataDict, lis)
saveResults(model, dataDict, lis)

