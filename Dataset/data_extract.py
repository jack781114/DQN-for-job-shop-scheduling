import pickle
import os

def change(A,B):
    f=r'.\Dataset'
    try:
        file=os.path.join(f,A,A+str(B)+'.pkl')
        with open(file,'rb') as f:
            d= pickle.load(f)
    except:
        file = os.path.join(f, A, A + '0' + str(B) + '.pkl')
        with open(file,'rb') as f:
            d= pickle.load(f)
    n, m, M = d[0], d[1], d[2]
    PT = []
    MT = []
    for i in range(len(M)):
        L = []
        L1 = []
        for j in range(0, len(M[i]), 2):
            L.append(M[i][j])
            L1.append(M[i][j + 1])
        MT.append(L)
        PT.append(L1)

    return n,m,PT,MT

# n,m,PT,MT=change('la',1)
# print(n,m,PT,MT)
