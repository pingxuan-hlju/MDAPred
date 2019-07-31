from numpy import *
import csv
import matplotlib
import sys
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc

MS = loadtxt('miRNA_sim.txt')                       #(492, 492)
DS = loadtxt("Dis_Sim_T.txt")                      #(329, 329)
A = loadtxt("mi_dis.txt")            #(492, 329)
Family = loadtxt("C_family3.txt")  #(492, 58)
Cluster = loadtxt("C_cluster3.txt")#(492, 570)
C = hstack((Family,Cluster))                     #(492, 628)

n=200
X1 = random.uniform(0,1,(shape(MS)[0],n))
X2 = random.uniform(0,1,(shape(DS)[0],n))
X3 = random.uniform(0,1,(shape(C)[1],n))
U = random.uniform(0,1,(shape(A)[0],shape(A)[1]))

def K_neighbor(S1,k):
    S = zeros(shape(S1))
    for i in range (shape(S1)[0]):
        sort_Data = sorted(enumerate(S1[i]), key=lambda x: x[1], reverse= True)  #已经排序的带有下表和元素值的列表
        for j in range(k):
            S[i][sort_Data[j][0]] = 1                                          #将对应的k个最近的邻居设为1，其余为0
        for j1 in range(shape(S1)[1]):
            S1[j1][i]=S1[i][j1]
    return S
###################################################  拉普拉斯正则化###################################################
def lapulace(S):
    D = zeros(shape(S))                                                       #D矩阵是S矩阵的每行的和放在主对角线上
    for i in range(shape(S)[0]):
        t = sum(S[i])
        D[i][i] = t
    return D
#############################################更新U##########################################
def updateU(X1,X2,X3,A1,U,MS,DS,C):
    B1 = ones(shape(A1))
    S1 = K_neighbor(MS, 10)
    D1 = lapulace(S1)
    # S2 = K_neighbor(DS, 10)
    # D2 = lapulace(S2)
    Y = A1/1

    r1 =A1
    # r1 = multiply(Y,A1)
    r2 = a1*(MS.dot(X1).dot(X2.T).dot(DS.T))
    r3 = a2 * (C.dot(X3).dot(X2.T).dot(DS.T))
    r4 = a3 * (S1.dot(U))
    # r5 = 2*a4 *(U.dot(S2))
    Num = 2 * r1 + 2 * r2 + 2 * r3 + 2*r4
    # t1 = multiply(Y,U)
    t1=U
    t23 = (U)
    t4 = a3 * (D1.dot(U))
    # t5 = 2*a4 * (U.dot(D2))
    t6 = a5 * B1
    Deno =2*t1+ 2 * (a1 + a2) * t23 + 2*t4  + t6
    Unew = multiply(divide(Num, Deno), U)
    return Unew
#######################################################更新X1###############################
def updateX1(X1,X2,U,MS,DS):
    r11 =((MS.T).dot(U).dot(DS).dot(X2))
    t11 = ((MS.T).dot(MS).dot(X1).dot(X2.T).dot(DS.T).dot(DS).dot(X2))
    z11 = divide(r11, t11)
    X1new = multiply(X1, z11)
    return X1new
######################################################更新 X2########################################
def updateX2(X1,X2,X3,U,MS,DS,C):
    r22 = a1 * ((DS.T).dot(U.T).dot(MS).dot(X1)) + a2 * ((DS.T).dot(U.T).dot(C).dot(X3))
    t22 = a1 * ((DS.T).dot(DS).dot(X2).dot(X1.T).dot(MS.T).dot(MS).dot(X1)) + a2 * (
        (DS.T).dot(DS).dot(X2).dot(X3.T).dot(C.T).dot(C).dot(X3))
    z22 = divide(r22, t22)
    X2new = multiply(X2, z22)
    return X2new
#######################################################更新X3########################################
def updateX3(X2,X3,U,DS,C):
    r33 = ((C.T).dot(U).dot(DS).dot(X2))
    t33 = ((C.T).dot(C).dot(X3).dot(X2.T).dot(DS.T).dot(DS).dot(X2))
    z33 = divide(r33, t33)
    X3new = multiply(X3, z33)
    return X3new
######################################################迭代##########################################
def iteration(X1,X2,X3,A1,U,MS,DS,C):

    S1 = K_neighbor(MS, 10)
    D1 = lapulace(S1)
    # S2 = K_neighbor(DS, 10)
    # D2 = lapulace(S2)

    for i in range(50):
        Unew = updateU(X1,X2,X3,A1,U,MS,DS,C)
        X1new = updateX1(X1,X2,Unew,MS,DS)
        X2new = updateX2(X1new,X2,X3,Unew,MS,DS,C)
        X3new = updateX3(X2new,X3,Unew,DS,C)
        y1 = linalg.norm((U - A1)) ** 2
        y2=a1*linalg.norm((U-MS.dot(X1).dot(X2.T).dot(DS.T)))**2
        y3=a2*linalg.norm((U-C.dot(X3).dot(X2.T).dot(DS.T)))**2
        y4=a3*trace(((U.T).dot(D1-S1).dot(U)))
        # y5=a4*trace((U.dot(D2-S2).dot(U.T)))
        y6=a5*linalg.norm((U),ord=1)
        L= y1+y2+y3+y4+y6
        U=Unew
        X1=X1new
        X2=X2new
        X3=X3new
        # if(i%20==0):
        #     print(L)
        # savetxt("u!!.txt",U)
    return Unew
################################################   计算有效行  ##############################################
def count_valid_data(U):
    f=zeros((U.shape[0],1),dtype=float64)
    for i in range(U.shape[0]):
        f[i] = sum(U[i] > 0)##RD[i]为该行的每个元素
    return f
###############################################
def caculate_TPR_FPR(RD, f, B):
    old_id = argsort(-RD)  # 记录排序前的位置
    min_f = int(min(f))  # 最小的有效数据的数目
    max_f = int(max(f))
    print('列数', min_f)
    TP_FN = zeros((RD.shape[0], 1), dtype=float64)  # 真正例总数
    FP_TN = zeros((RD.shape[0], 1), dtype=float64)  # 真反例总数
    TP = zeros((RD.shape[0], max_f), dtype=float64)  # 正例被判断为正例的个数
    TP2 = zeros((RD.shape[0], min_f), dtype=float64)
    FP = zeros((RD.shape[0], max_f), dtype=float64)  # 假例被判断为正例的个数
    FP2 = zeros((RD.shape[0], min_f), dtype=float64)
    P = zeros((RD.shape[0], max_f), dtype=float64)  # 查准率
    P2 = zeros((RD.shape[0], min_f), dtype=float64)  # 查准率

    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)

    for i in range(RD.shape[0]):
        # kk = f[i] / min_f  # 这行数据的有效数据数目是最小数目的多少
        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:  # j*kk+(kk-1) 按比例抽样
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)


    ki = 0  # 无效行的数目
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]  # 此时的TP里其实装的是TPR
            FP[i] = FP[i] / FP_TN[i]

    for i in range(RD.shape[0]):
        kk = f[i] / min_f  # 这行数据的有效数据数目是最小数目的多少倍
        for j in range(min_f):
            TP2[i][j] = TP[i][int(round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(round_(((j + 1) * kk))) - 1]

    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    P = P2.sum(0) / (P.shape[0] - ki)
    # print('行数', TP.shape[0] - ki)
    # print(FP2[:,FP2.shape[1]-1])
    # print(sum(FP2[:,FP2.shape[1]-1] == 0))
    # # print(ki)
    # print(sum(FP2[:, FP2.shape[1] - 1] == 1))

    # print(FPR.shape)
    # print(TPR.shape)
    return TPR, FPR, P

def curve(TPR,FPR,P):

    plt.figure()
    plt.subplot(121)
    # plt.xlim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    plt.title("ROC curve  (AUC = %.4f)" % (auc(FPR, TPR)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(FPR, TPR)
    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title('PR curve (AUC = %.4f)'%(auc(TPR,P)+(TPR[0]*P[0])))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(TPR,P)
    print("画图")
    plt.show()
def sset_ttt(U1,B):
    for i in range(793):
        for j in range(341):
            if B[i][j]==1:
                U1[i][j] = -1
    return U1
def cross_vaildtion(A):

    # for i in range(fold):                         #对每次交叉验证
    mir_names = []
    a = loadtxt("mi_dis.txt")
    print(a.shape)
    with open("mirna_sort793.txt", 'r') as f:
        for line in f:
            mir_names.append(line.strip('\n').split(',')[0])
    B=A/1                                    #相当于deepcopy
    # A1 = set_test1(A,S_data,i)                 #训练集部分置0
    U1 = iteration(X1,X2,X3,A,U,MS,DS,C)
    # U1, B = set_train(U1, B, i, fold)  # 将训练集的四份置-1
    U1 = sset_ttt(U1,B)
    # savetxt("U_all.txt",U1)

    U1 = U1.T
    breast_cancer = U1[0]
    lung_cancer = U1[9]
    pancreatic_cancer = U1[17]

    breast_cancer = array(breast_cancer)
    lung_cancer = array(lung_cancer)
    pancreatic_cancer = array(pancreatic_cancer)

    # print("breast_cancer data :",breast_cancer, '\n')
    old_id = argsort(-breast_cancer)
    # print("*******", old_id)
    # print("======", old_id[:50])
    breast_name = []
    for i in range(50):
        id = old_id[i]
        breast_name.append(mir_names[id],)
    #print(breast_name)
    savetxt("breast_cancer.txt",breast_name,fmt="%s",newline='\n')

    # print("lung_cancer data :",lung_cancer, '\n')
    old_id = argsort(-lung_cancer)
    # print("*******", old_id)
    # print("======", old_id[:50])
    lung_name = []
    for i in range(50):
        id = old_id[i]
        lung_name.append(mir_names[id])
    # print(lung_name)
    savetxt("lung_cancer.txt", lung_name,fmt="%s", newline='\n')

    # print("lung_cancer data :",lung_cancer, '\n')
    old_id = argsort(-pancreatic_cancer)
    print("*******", old_id)
    print("======", old_id[:50])
    pancreatic_name = []
    for i in range(50):
        id = old_id[i]
        pancreatic_name.append(mir_names[id])
    # print(lung_name)
    savetxt("pancreatic_cancer.txt", pancreatic_name,fmt="%s", newline='\n')

    # f = count_valid_data(U1) # 每一行的有效数据
    # f = count_num(B)
    # U1 = U1.T
    # B = B.T
    # TPR, FPR, P = caculate_TPR_FPR(U1, f, B)  # 计算TPR,FPR，P
    # TPR,FPR,P = compute_fpr(U1,B,f)
    # TPR15,FPR15,P15 = commondis15(FPR,TPR,P)
    # print("TPR,FPR:",np.shape(TPR),np.shape(FPR))
    # roc_auc,pr_auc = curve(TPR, FPR, P)  # 画图
    # print("*-*-*-*-*",roc_auc,pr_auc)
    # if fold == 1:
    #     with open('parameter','a',newline='') as csvfiles:
    #         writer = csv.writer(csvfiles)
    #         infomation = [a1,a2,a3,a4,a5,'结果为',roc_auc,pr_auc]
    #         writer.writerow(infomation)
    # plt.show()

if __name__ == '__main__':
    fold = 5
    # e = [0.1,10]
    a1,a2,a3,a4,a5=0.1,0.1,0.1,0.1,0.1

    cross_vaildtion(A)
    print("11111111111")

