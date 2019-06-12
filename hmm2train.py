#!/usr/bin/env python3

import re
import numpy as np

class Family():

    def __init__(self,filename,mode='train'):
        with open(filename,"r") as f:
            data = f.readlines()
        self.seq = []
        self.sta = []

        for i in range(len(data)):
            if data[i][0] != "#" and '.' in data[i]:
                splitted = re.split(" ",data[i])
                temp = splitted[-1]
                self.seq.append(temp[:-1])
                self.sta.append([])
                for j in range(len(self.seq[-1])):
                    self.sta[-1].append('G'+str(j) if self.seq[-1][j] == '.' else 'M'+str(j))

        split = len(self.seq)//2
        if mode =='train':
            self.seq = self.seq[:split]
            self.sta = self.sta[:split]
        else:
            self.seq = self.seq[split:]
            self.sta = self.sta[split:]

    
    def __len__(self):
        return len(self.seq)
               
def train2hmm(family):
    '''
    :family: __main__.Family
    :r:probs: numpy.ndarray
    :r:A1: dict{str:numpy.ndarray}
    :r:A2: dict{(str,str):numpy.ndarray}
    :r:B1: dict{str:numpy.ndarray}
    :r:B2: dict{(str,str):numpy.ndarray}
    '''
    aminoList = 'MRPGCNWTYFHAKIVDSLEQ.' #len==21
    seqLen = len(family.seq[0])
    groupLen = len(family)
    probs_ = np.zeros((seqLen,21))
    for seq in family.seq:
        for j,site in enumerate(seq):
            probs_[j,aminoList.index(site)] += 1
    probs = probs_[:,0:20]
    probs = probs.sum(axis=0)
    probs /= probs.sum()

    #2
    A1 = dict()
    for sta_seq in family.sta:
        for j in range(seqLen-1):
            if sta_seq[j] not in A1:
                A1[sta_seq[j]] = dict()
            if sta_seq[j+1] not in A1[sta_seq[j]]:
                A1[sta_seq[j]][sta_seq[j+1]] = 1
            else:
                A1[sta_seq[j]][sta_seq[j+1]] += 1
    else:
        for i in A1.keys():
            temp = sum(A1[i].values())
            for j in A1[i].keys():
                A1[i][j] /= temp
    A2 = dict()
    for sta_seq in family.sta:
        for j in range(1,seqLen-1):
            if (sta_seq[j],sta_seq[j-1]) not in A2:
                A2[(sta_seq[j],sta_seq[j-1])] = dict()
            if sta_seq[j+1] not in A2[(sta_seq[j],sta_seq[j-1])]:
                A2[(sta_seq[j],sta_seq[j-1])][sta_seq[j+1]] = 1
            else:
                A2[(sta_seq[j],sta_seq[j-1])][sta_seq[j+1]] += 1
    else:
        for i in A2.keys():
            temp = sum(A2[i].values())
            for j in A2[i].keys():
                A2[i][j] /= temp
                
    #3
    B1 = dict()
    for i in range(groupLen):
        seq = family.seq[i]
        sta_seq = family.sta[i]
        for j in range(seqLen):
            if sta_seq[j][0]!='G':
                if sta_seq[j] not in B1:
                    B1[sta_seq[j]] = np.zeros(20)
                B1[sta_seq[j]][aminoList.index(seq[j])] += 1
    else:
        for i in B1.keys():
            B1[i] /= B1[i].sum()
    B2 = dict()
    for i in range(groupLen):
        seq = family.seq[i]
        sta_seq = family.sta[i]
        for j in range(1,seqLen):
            if sta_seq[j][0]!='G':
                if (sta_seq[j],sta_seq[j-1]) not in B2:
                    B2[(sta_seq[j],sta_seq[j-1])] = np.zeros(20)
                B2[(sta_seq[j],sta_seq[j-1])][aminoList.index(seq[j])] += 1
    else:
        for i in B2.keys():
            B2[i] /= B2[i].sum()

    return probs,A1,A2,B1,B2

    



if __name__ == "__main__":
    '''
    This is to handle loading the input dataset, run functions, and output the results
    '''
    # parse user arguments
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--family', required=True, type=str, default='stdin', help="Input Gene Family")
    parser.add_argument('-o', '--output', required=False, type=str, default='stdout', help="Output Score")
    args = parser.parse_args()

    # load gene family
    try:
        family_train = Family(args.family,mode='train')
        family_test = Family(args.family,mode='test')
    except:
        raise ValueError("Wrong direction or file!")

    # train the 2hmm
    probs,A1,A2,B1,B2 = train2hmm(family_train)

    # evaluate the 2hmm


    #output score
    outfile = open(args.output,'w')
    outfile.write(f'{len(probs)} {len(A1)} {len(A2)} {len(B1)} {len(B2)}')
    outfile.close()