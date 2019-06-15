# ECE208_HMM2

Guangjun Xue A53283032\
Jiangrui Chen A53281208\
Linfeng Chen A53270085\
Tianming Sun A53267707

## Dataset
Five families from a clan in PFam Library are given:\
https://pfam.xfam.org/clan/PAN 

Family1: PAN_1 (PF00024)\
Family2: PAN_2 (PF08276)\
Family3: PAN_3 (PF08277)\
Family4: PAN_4 (PF14295)\
Family5: MANEC (PF07502)

## Input
python3 ece208_2hmm.py -f 'dataset/1.txt,dataset/2.txt,dataset/3.txt,dataset/4.txt,dataset/5.txt' -m 1 -o 1-order.txt\
python3 ece208_2hmm.py -f 'dataset/1.txt,dataset/2.txt,dataset/3.txt,dataset/4.txt,dataset/5.txt' -m 2 -o 2-order.txt
