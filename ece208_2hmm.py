#!/usr/bin/env python3

import re
import numpy as np

class Family():

    def __init__(self,filename,mode='train'):
        with open(filename,"r") as f:
            data = f.readlines()
        self.seq = []
        self.sta = []
        self.adj_seq = []
        self.modLen = 0

        for i in range(len(data)):
            if data[i][0] != "#" and '.' in data[i]:
                splitted = re.split(" ",data[i])
                temp = splitted[-1]
                self.seq.append(temp[:-1])
        
        split = len(self.seq)//2
        if mode =='train':
            self.seq = self.seq[:split]
        else:
            self.seq = self.seq[split:]
        
        np.random.shuffle(self.seq)
        seqLen = len(self.seq[0])
        self.seqLen = seqLen

        gapCount = np.zeros(seqLen)
        for item in self.seq:
            for j in range(seqLen):
                if item[j]=='.':
                    gapCount[j] +=1
        gapCount /= len(self.seq)
        templete = []
        MCount = 0
        for j in range(seqLen):
            if gapCount[j]<0.5:
                MCount += 1
                temp = 'M'+str(MCount)
                templete.append(temp)
            else:
                temp = 'I'+str(MCount)
                templete.append(temp)
        else:
            self.modLen = MCount

        for i in range(len(self.seq)):
            self.sta.append([])
            self.adj_seq.append('')
            for j in range(seqLen):
                if self.seq[i][j]=='.' and gapCount[j]<0.5:
                    temp='D'+templete[j][1:]
                    self.sta[i].append(temp)
                    self.adj_seq[i] += self.seq[i][j]
                elif self.seq[i][j]=='.' and gapCount[j]>=0.5:
                    pass
                else:
                    self.sta[i].append(templete[j])
                    self.adj_seq[i] += self.seq[i][j]
    
    def __len__(self):
        return len(self.seq)
               
def train2hmm(family):
    '''
    :family: __main__.Family
    :r:probs: numpy.ndarray
    :r:A1: numpy.ndarray((family.modLen+1,3,3))
    :r:A2: numpy.ndarray((family.modLen+1,3,3,3))
    :r:B1: numpy.ndarray((3*(family.modLen+1),22))
    :r:B2: numpy.ndarray((family.modLen+1,3,3,22))
    '''
    aminoList = 'MRPGCNWTYFHAKIVDSLEQX.' #len==22
    seqLen = len(family.seq[0])
    groupLen = len(family)
    b = 0.1

    #1
    probs_ = np.zeros((seqLen,22))
    for seq in family.seq:
        for j,site in enumerate(seq):
            probs_[j,aminoList.index(site)] += 1
    probs = probs_[:,0:21]
    probs = probs.sum(axis=0)
    probs /= probs.sum()


    #2 States Transition
    A1 = np.zeros((family.modLen+1,3,3))
    ax = {'M':0,'D':1,'I':2}
    for sta_seq in family.sta:
        for j in range(len(sta_seq)-1):
            t = int(sta_seq[j][1:])
            A1[t,ax[sta_seq[j][0]],ax[sta_seq[j+1][0]]] += 1
    else:
        for k in range(family.modLen+1):
            for i in range(3):
                A1[k,i,:] += b
                A1[k,i,:] /= A1[k,i,:].sum()

    A2 = np.zeros((family.modLen+1,3,3,3))
    ax = {'M':0,'D':1,'I':2}
    for sta_seq in family.sta:
        for j in range(1,len(sta_seq)-1):
            t = int(sta_seq[j][1:])
            A2[t,ax[sta_seq[j][0]],ax[sta_seq[j-1][0]],ax[sta_seq[j+1][0]]] += 1
    else:
        for k in range(family.modLen+1):
            for i in range(3):
                for j in range(3):
                    A2[k,i,j,:] += b
                    A2[k,i,j,:] /= A2[k,i,j,:].sum()
                
    #3 Emission Probabilities
    B1 = np.zeros((family.modLen+1,3,22))
    ax = {'M':0,'D':1,'I':2}
    for i in range(groupLen):
        sta_seq = family.sta[i]
        adj_seq = family.adj_seq[i]
        for j in range(len(sta_seq)):
            t = int(sta_seq[j][1:])
            B1[t,ax[sta_seq[j][0]],aminoList.index(adj_seq[j])] += 1
    else:
        for k in range(family.modLen+1):
            for i in range(3):
                if i==1:
                    B1[k,i,:] = 0
                    B1[k,i,-1] = 1
                else:
                    B1[k,i,:-1] += b
                    B1[k,i,:-1] /= B1[k,i,:].sum()

    B2 = np.zeros((family.modLen+1,3,3,22))
    ax = {'M':0,'D':1,'I':2}
    for i in range(groupLen):
        sta_seq = family.sta[i]
        adj_seq = family.adj_seq[i]
        for j in range(1,len(sta_seq)):
            t = int(sta_seq[j][1:])
            B2[t,ax[sta_seq[j][0]],ax[sta_seq[j-1][0]],aminoList.index(adj_seq[j])] += 1
    else:
        for k in range(family.modLen+1):
            for i in range(3):
                for j in range(3):
                    if j==1:
                        B2[k,i,j,:] = 0
                        B2[k,i,j,-1] = 1
                    else:
                        B2[k,i,j,:-1] += b
                        B2[k,i,j,:-1] /= B2[k,i,j,:].sum()

    return probs,A1,A2,B1,B2

def nextsta(s,modeLen):
    #ax = {'M':0,'D':1,'I':2}
    if len(s)==2:
        if s[0]<modeLen:
            return [(s[0]+1,0),(s[0]+1,1),(s[0],2)]
        else:
            return [(s[0],2)]
    else:
        if s[2]<modeLen:
            return [(s[2],s[3],s[2]+1,0),(s[2],s[3],s[2]+1,1),(s[2],s[3],s[2],2)]
        else:
            return [(s[2],s[3],s[2],2)]
        
def forward(seq,A,B,mode=1):
    seqLen = len(seq)
    modeLen = A.shape[0]-1
    #ax = {'M':0,'D':1,'I':2}
    aminoList = 'MRPGCNWTYFHAKIVDSLEQX.' # len=22
    
    if mode == 1: # 1-order hmm
        cur_staPool = [(1,0),(1,1)]#['M1','D1'] t=1
        f = np.array([0.8,0.2])
        for i in range(seqLen-1):
            f = f/f.sum() if f.sum() != 0 else f * 0
            cur_staPool_ = []
            for ind,item in enumerate(f):
                if item>0:
                    cur_staPool_.append(cur_staPool[ind])
            cur_staPool = cur_staPool_.copy()
            ind_del = np.where(f==0)
            f = np.delete(f, ind_del)
            #print(i,seqLen,len(cur_staPool),f.sum())
            # find all possible next states
            next_staPool = []
            for item in cur_staPool:
                next_staPool.extend(nextsta(item,modeLen))
            next_staPool = list(set(next_staPool))

            # apply transimition and emission possibilities
            temp1 = np.zeros(len(next_staPool))
            for k,sta1 in enumerate(next_staPool):
                temp2 = np.zeros(len(cur_staPool))
                for j,sta2 in enumerate(cur_staPool):
                    if sta1 in nextsta(sta2,modeLen):
                        temp2[j] = A[sta2[0],sta2[1],sta1[1]]
                    else:
                        temp2[j] = 0
                temp2 = temp2 * f
                temp1[k] = B[sta1[0],sta1[1],aminoList.index(seq[i+1])] * temp2.sum()
            #temp1 is f_k(i+1)
            cur_staPool = next_staPool.copy()
            #f = temp1.copy() *10
            f = temp1.copy()
                                   
    else:# 2-order hmm
        cur_staPool = [(1,0,2,0),(1,1,2,0),(1,0,2,1),(1,1,2,1),(1,0,1,2),(1,1,1,2)]#['M1M2','D1M2','M1D2','D1D2','M1I1','D1I1'] 
        f = np.array([0.3,0.2,0.2,0.1,0.1,0.1])
        for i in range(1,seqLen-1):
            f = f/f.sum() if f.sum() != 0 else f * 0
            cur_staPool_ = []
            for ind,item in enumerate(f):
                if item>0:
                    cur_staPool_.append(cur_staPool[ind])
            cur_staPool = cur_staPool_.copy()
            ind_del = np.where(f==0)
            f = np.delete(f, ind_del)
            #print(i,seqLen,len(cur_staPool),f.sum())
            # find all possible next states
            next_staPool = []
            for item in cur_staPool:
                next_staPool.extend(nextsta(item,modeLen))
            next_staPool = list(set(next_staPool))

            # apply transimition and emission possibilities
            temp1 = np.zeros(len(next_staPool))
            for k,sta1 in enumerate(next_staPool):
                temp2 = np.zeros(len(cur_staPool))
                for j,sta2 in enumerate(cur_staPool):
                    if sta1 in nextsta(sta2,modeLen):
                        temp2[j] = A[sta2[2],sta2[3],sta2[1],sta1[3]] * B[sta1[2],sta1[3],sta2[3],aminoList.index(seq[i+1])]
                    else:
                        temp2[j] = 0
                temp2 = temp2 * f
                temp1[k] = temp2.sum()
            #temp1 is f_k(i+1)
            cur_staPool = next_staPool.copy()
            #f = temp1.copy() *10
            f = temp1.copy()
    return f.sum()

def evaluate(A,B,testcase,j,mode=1):
    testres = []
    for i in range(len(A[0])):
        ps = []
        for ind,seq in enumerate(testcase.seq):
            print(ind)
            ps.append(forward(seq,A[mode-1][i],B[mode-1][i],mode=mode))
    
        testres.append(ps)
    testres_ind = np.argmax(np.array(testres),axis = 0)
    testacc = np.mean(np.equal(testres_ind,[j] * len(testres_ind)).astype('float32'))

    return testacc

if __name__ == "__main__":
    '''
    This is to handle loading the input dataset, run functions, and output the results
    '''
    # parse user arguments
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--family', required=True, type=str, default='stdin', help="Input Gene Family")#type in like"--family seed1.txt,seed2.txt,seed3.txt"
    parser.add_argument('-o', '--output', required=False, type=str, default='stdout', help="Output Score")
    parser.add_argument('-m', '--mode', required=True, type=int, default=1, help="Mode")
    args = parser.parse_args()

    #suppose there are five families and you need to predict the label of testset
    family = re.split(",",args.family)
    testset = []
    A,B = [[],[]], [[],[]]
    for i in range(5):
        print(f'==={i}===')
        family_train = Family(family[i],mode='train')
        family_test = Family(family[i],mode='test')
        probs,A1,A2,B1,B2 = train2hmm(family_train)
        A[0].append(A1)
        A[1].append(A2)
        B[0].append(B1)
        B[1].append(B2)
        testset.append(family_test)

    testacc = []#results should like [0.8,0.9,0.6,0.6,0.6]
    for j in range(5):#for different family testset, evaluate over all models
        testacc.append(evaluate(A,B,testset[j],j,mode=args.mode))
    print(testacc)

    testacc_w = ''
    for item in testacc:
        testacc_w += ' '+str(item)
    outfile = open(args.output,'w')
    outfile.write(testacc_w.strip())
    outfile.close()
