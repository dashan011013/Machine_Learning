# -*- coding: utf-8 -*-
import numpy as np

def AnalysisData(X,DataType=None):
    # 分析数据
    # DataType 返回每个属性的数据类型，若为字符串类型，说明是离散型，并统计各个可能取值
    # X1 返回数字化的数据，将离散型数值转变为相应取值的编号
    m=len(X)     #样本数
    n=len(X[0])  #属性数
    if DataType==None: #如果未从外部输入DataType,则需自动创建
        DataType=[]
        for p in range(n):
            DataType.append([type(X[0][p])])
            if DataType[p][0]==str:
                for q in range(m):
                    if X[q][p] not in DataType[p]:
                        DataType[p].append(X[q][p])
    X1=[]   #数字化的数据
    for p in range(m):
        temp=[]
        for q in range(n):
            if DataType[q][0]==str:
                temp.append(DataType[q].index(X[p][q]))
            else:
                temp.append(X[p][q])
        X1.append(temp)
    return DataType,X1

def most(Y):
    # 统计样本集中的多数类
    if len(Y)==0:
        return None
    Yvalue=[]  #Y取值
    Ynum=[]    #取值数目
    for y in Y:
        if y not in Yvalue:
            Yvalue.append(y)
            Ynum.append(1)
        else:
            Ynum[Yvalue.index(y)]+=1
    return Yvalue[Ynum.index(max(Ynum))]

def Index(Y,rule='InfoGain'):
    # 计算信息熵或者基尼系数
    if len(Y)==0:
        if rule=='InfoGain':
            return 0
        elif rule=='Gini':
            return 1
    Yvalue=[]  #Y取值
    Ynum=[]    #取值数目
    for y in Y:
        if y not in Yvalue:
            Yvalue.append(y)
            Ynum.append(1)
        else:
            Ynum[Yvalue.index(y)]+=1
    pk=np.array(Ynum)/sum(Ynum)
    if rule=='InfoGain':
        return sum(-pk*np.log2(pk))
    elif rule=='Gini':
        return 1-sum(pk**2)

def score(X,Y,considerfeature,DataType,rule='InfoGain'):
    # 这里considerfeature为单个考察属性的索引号
    # 仅适用于信息熵和基尼系数两种选择规则，哪种规则由参数rule确定
    # 由于计算信息熵和基尼值的过程多有相似，因此统一编程
    # 返回信息熵/基尼值
    # 若为连续属性，还要返回划分点
    f=considerfeature
    if DataType[f][0]==str:  #若该特征取值为离散型
        Gain=Index(Y) #信息增益初始值
        Gini=0
        for v in range(1,len(DataType[f])):
            Yv=Y[X[:,f]==v]
            Gain-=Index(Yv)*len(Yv)/len(Y)
            Gini+=Index(Yv,'Gini')*len(Yv)/len(Y)
        Gain=[Gain]
        Gini=[Gini]
    else:   #若该特征取值为连续型
        Gain=[]
        Gini=[]
        values=np.array(list(set(X[:,f])))  #利用集合去除重复取值
        if len(values)==1:  #若当前子集X中该特征取值相同
            Gain=[0,values]
            Gini=[Index(Y,'Gini'),values]
        else:
            t=(np.sort(values)[:-1]+np.sort(values)[1:])/2  #候选划分点
            for tt in t:
                Y1=Y[X[:,f]<=tt]; Y2=Y[X[:,f]>tt]
                Gain.append(Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y))
                Gini.append(Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y))
            Gain=[max(Gain),t[Gain.index(max(Gain))]]
            Gini=[min(Gini),t[Gini.index(min(Gini))]]
    if rule=='InfoGain':
        return Gain
    if rule=='Gini':
        return Gini


def perturb(X,Y,w,p,rule,stagnant):
    # 确定性扰动
    m,n=X.shape
    X1=np.c_[X,np.ones([m,1])]  #增加全1列
    #--------原始w的划分结果-------
    Y1=Y[np.dot(X1,w)<=0]
    Y2=Y[np.dot(X1,w)>0]
    Gain0=Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y)
    Gini0=Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y)
    #--------最佳的wp更新值-------
    U=w[p]-np.dot(X1,w)/X1[:,p]
    U=np.array(list(set(U)))               #利用集合去除重复取值
    wp=(np.sort(U)[:-1]+np.sort(U)[1:])/2  #二分法确定候选划分点
    Gain=[]
    Gini=[]
    for t in wp:
        w_temp=w.copy()
        w_temp[p]=t
        Y1=Y[np.dot(X1,w_temp)<=0]; Y2=Y[np.dot(X1,w_temp)>0]
        Gain.append(Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y))
        Gini.append(Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y))
    Gain=[max(Gain),wp[Gain.index(max(Gain))]]
    Gini=[min(Gini),wp[Gini.index(min(Gini))]]
    #--------更新wp-------
    if rule=='InfoGain':
        if Gain[0]>Gain0:
            w[p]=Gain[1]
            return 0
        elif (Gain[0]==Gain0)and(np.random.rand()<np.exp(-stagnant)) :
            w[p]=Gain[1]
            return stagnant+1
        else:
            return stagnant
    if rule=='Gini':
        if Gini[0]<Gini0:
            w[p]=Gini[1]
            return 0
        elif (Gini[0]==Gini0)and(np.random.rand()<np.exp(-stagnant)) :
            w[p]=Gain[1]
            return stagnant+1
        else:
            return stagnant

def rand_perturb(X,Y,w,randw,rule):
    # 随机性扰动
    m,n=X.shape
    X1=np.c_[X,np.ones([m,1])]  #增加全1列
    #--------原始w的划分结果-------
    Y1=Y[np.dot(X1,w)<=0]
    Y2=Y[np.dot(X1,w)>0]
    Gain0=Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y)
    Gini0=Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y)
    #--------最佳扰动系数k-------
    U=-np.dot(X1,w)/np.dot(X1,randw)
    U=np.array(list(set(U)))               #利用集合去除重复取值
    k=(np.sort(U)[:-1]+np.sort(U)[1:])/2  #二分法确定候选划分点
    Gain=[]
    Gini=[]
    for kk in k:
        w_temp=w+kk*randw
        Y1=Y[np.dot(X1,w_temp)<=0]; Y2=Y[np.dot(X1,w_temp)>0]
        Gain.append(Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y))
        Gini.append(Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y))
    Gain=[max(Gain),k[Gain.index(max(Gain))]]
    Gini=[min(Gini),k[Gini.index(min(Gini))]]
    #--------返回结果-------
    if rule=='InfoGain':
        if Gain[0]>Gain0:
            return w+Gain[1]*randw,True
        else:
            return w,False
    if rule=='Gini':
        if Gini[0]<Gini0:
            return w+Gini[1]*randw,True
        else:
            return w,False
            
def compare(X,Y,w1,w2,rule):
    # 比较两个组合系数哪个好
    m,n=X.shape
    X1=np.c_[X,np.ones([m,1])]  #增加全1列
    #--------w1的划分结果-------
    Y1=Y[np.dot(X1,w1)<=0]
    Y2=Y[np.dot(X1,w1)>0]
    Gain1=Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y)
    Gini1=Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y)
    #--------w2的划分结果-------
    Y1=Y[np.dot(X1,w2)<=0]
    Y2=Y[np.dot(X1,w2)>0]
    Gain2=Index(Y)-Index(Y1)*len(Y1)/len(Y)-Index(Y2)*len(Y2)/len(Y)
    Gini2=Index(Y1,'Gini')*len(Y1)/len(Y)+Index(Y2,'Gini')*len(Y2)/len(Y)
    #--------返回结果-------
    if rule=='InfoGain':
        if Gain2>Gain1:
            return w2,Gain2
        else:
            return w1,Gain1
    if rule=='Gini':
        if Gini2<Gini1:
            return w2,Gini2
        else:
            return w1,Gini1   

def OC1(X,Y,w0=None,rule='Gain'):
    # OC1算法寻找最佳组合系数
    m,n=X.shape  # 样本数和特征数
    iteration=25  # 尝试iteration次的重新初始化
    repeat=50    # 采用R-50方案来对不同参数进行更新
    for i in range(iteration):
        if (w0!=None) and(i==0):
            w=w0.copy()
        else:
            w=np.random.randn(n+1)          # 随机初始化参数
        again=True
        while again:
            #--------确定性扰动--------
            stagnant=0
            for r in range(repeat):
                p=np.random.randint(n+1)    # 随机选择一个参数来更新
                stagnant=perturb(X,Y,w,p,rule,stagnant)
                # 注意：w是一个向量，传入函数后，若对其改变，
                # 原向量w也将随之改变，这在编程中很容易出错，
                # 而这里则利用该陷阱
            #--------随机性扰动--------
            randw=np.random.randn(n+1)   # 随机扰动方向
            w,again=rand_perturb(X,Y,w,randw,rule)
        if i==0:
            bestw=w.copy()
            continue
        bestw,bestvalue=compare(X,Y,bestw,w,rule)
    return bestvalue,bestw

def choose(X,Y,FeatureName,considerfeature,DataType,rule='InfoGain',equalchoose=False):
    # 选择最优划分方式
    # 首先对所有属性进行单变量划分
    # 然后对连续性属性进行多变量划分，采用OC1算法
    # 最后择其最优者作为最终划分方式

    scores=[] # 下面的values用于存储各个属性划分下的纯度值 (信息增益、基尼系数)
              # scores除了存储这些指标，还有划分点信息 (对于连续取值属性)

    #-------单变量划分-------
    for f in considerfeature:
        scores.append(score(X,Y,f,DataType,rule))
    values=[s[0] for s in scores]
    #----多变量划分(OC1)-----
    numf=[]        #数值型属性索引号
    numscores=[]   #数值型属性的划分得分和划分点
    for f in considerfeature:
        Type=DataType[f][0]
        if (Type==int)|(Type==float):
            numf.append(f)
            numscores.append(scores[considerfeature.index(f)])
    if len(numf)>=2:
        #-------最佳的单变量划分方式作为初始组合系数
        numvalues=[ns[0] for ns in numscores]
        index=numvalues.index(max(numvalues))
        w0=[0]*(len(numf)+1)
        w0[index]=1
        w0[-1]=-numscores[index][1]
        #--------由OC1算法得到的最佳纯度值和组合系数
        oc1_value,coes=OC1(X[:,numf],Y,w0.copy(),rule)  
        values.append(oc1_value)
    
    #-----最佳划分方式-------
    if rule=='InfoGain':
        bestvalue=max(values)
    if rule=='Gini':
        bestvalue=min(values)
    index=values.index(bestvalue)
    
    
    # return [numf,coes]   #直接返回多变量划分
    #-----返回结果-----------
    
    if index==len(scores):         #若为多变量划分
        return [numf,coes]
    elif len(scores[index])>1:     #若为连续型单属性划分
        return [considerfeature[index],scores[index][1]]
    else:                          #若为离散型单属性划分
        return [considerfeature[index]]

def CreatTree(Xt,Yt,FeatureName,LabelName,considerfeature='all',DataType=None,rule='InfoGain',cut=None,equalcut=True,deep=0,normal=None,lamuda=0.01,equalchoose=False,order=None):
    #   引入斜决策树算法OC1
    #================================================================
    #                      递归方法训练决策树
    # *****输入变量*****
    # Xt,Xv---------训练集和剪枝集的X---二层列表数据类型，列表内容可以是字符串或者数值
    # Yt,Yv---------训练集和剪枝集的X---单层列表数据类型，列表内容为数值
    # FeatureName---特征(属性)名称。----单层列表数据类型，列表内容为字符串                
    # LabelName-----类标记名称。--------单层字典类型，key为数值型，对应于Yt/Yv中所有可能数值，value为字符串
    # consi***ture--当前考察特征--------单层列表类型，内容为所考察特征所对应的索引号，默认为‘all’，意为所有特征
    # DataType------数据类型------------二层列表类型，格式为比如：[[type('str'),'a','b'],[type(0.0)]]
    #                                      意为有两个特征，第一个特征数据为字符串类型，可能取值为'a'和'b',
    #                                      第二个特征为数值型。
    # rule----------决策树划分规则------字符串类型，可能三种取值:'InfoGain'-信息熵,'Gini'-基尼,'LogisticReg'-对率回归
    # cut-----------剪枝类型------------字符串类型，可能三种取值：'pre'-预剪枝,'after'-后剪枝，None-不剪枝
    # equalcut------精度相等时是否剪枝---逻辑类型:True/False
    # deep----------当前决策树深度-------整数类型，初始值为0，(实际上相当递归深度)
    # normal--------归一化方式-----------字符串类型，三种取值：'min-max','z-score',None
    # lamuda--------规则化参数-----------数值类型，仅用于对率回归
    # equalchoose---是否手动选择属性-----逻辑类型:True/False，用于多个属性的信息熵/基尼值/权重值相等时
    # order---------各个属性的有序性-----单层列表类型，内容和True/False,表明各个属性的有序性
    # *****输出变量*****
    # tree----------决策树--------------多层嵌套字典类型
    #================================================================  
    #print(deep,considerfeature)
    #-------------初始化--------------
    if considerfeature=='all':   #初始时的'all'表示考虑所有特征
        considerfeature=list(range(len(FeatureName)))
    if deep==0:                  #初始时需要分析数据，将X数字化，并得到每个X的数据类型
        if order==None:
            order=[True]*len(FeatureName)
        m=len(Yt)
        DataType,Xt=AnalysisData(Xt,DataType)
        Xt=np.array(Xt)
        Yt=np.array(Yt)
    #------ --考察几种特殊情况---------
    if (Yt==Yt[0]).all():      #若所有样本属于同一类
        return LabelName[Yt[0]]
    if len(considerfeature)==0:  #若待考察特征为空
        return LabelName[most(Yt)]
    if (Xt[:,considerfeature]==Xt[0,considerfeature]).all():  #若所有样本取值相同
        return LabelName[most(Yt)]
    #--------最优划分属性-------------
    BestFeature=choose(Xt,Yt,FeatureName,considerfeature,DataType,rule,equalchoose)  #选择最佳划分属性
    bf=BestFeature[0]
    #------------分支-----------------
    if len(BestFeature)==1:  #若为单变量的离散型属性
        nt=FeatureName[bf]+'=?'  #nt意为nodetext
        tree={nt:{}}
        considerfeature.remove(bf)
        for k in range(1,len(DataType[bf])):
            Xtk=Xt[Xt[:,bf]==k,:]
            Ytk=Yt[Xt[:,bf]==k]
            if len(Ytk)==0:  #若该分支的训练集为空
                tree[nt][DataType[bf][k]]=LabelName[most(Yt)]
            else:
                tree[nt][DataType[bf][k]]=CreatTree(Xtk,Ytk,FeatureName,LabelName,considerfeature.copy(),DataType,rule,cut,equalcut,deep+1,normal,lamuda,equalchoose,order)
    elif type(bf)==int:     #若为单变量的连续型属性
        nt=FeatureName[bf]  #nt意为nodetext
        tree={nt:{}}
        t=BestFeature[1]    #划分点
        t=float('%.4f'%t)   # 这一句是考虑到这样一种情况：
                            # 比方t=0.999935,划分子集时，x=1将被划分到'x>t'
                            # 然而在决策树中存储的信息为t=1.0,见下面语句tree[nt]['<=%.3f'%t]=...
                            # 在后剪枝或者预测新样本时，x=1将被划分到'x<=t'分支
                            # 将会出现前后不一致的情况，甚至于出错
        #----------分支-----------
        Xt1=Xt[Xt[:,bf]<=t,:]; Yt1=Yt[Xt[:,bf]<=t]
        Xt2=Xt[Xt[:,bf]>t,:]; Yt2=Yt[Xt[:,bf]>t]
    
        if (len(Yt1)==0)|(len(Yt2)==0):
            tree=LabelName[most(Yt)]
        else:
            tree[nt]['<=%.4f'%t]=CreatTree(Xt1,Yt1,FeatureName,LabelName,considerfeature.copy(),DataType,rule,cut,equalcut,deep+1,normal,lamuda,equalchoose,order)
            tree[nt]['>%.4f'%t]=CreatTree(Xt2,Yt2,FeatureName,LabelName,considerfeature.copy(),DataType,rule,cut,equalcut,deep+1,normal,lamuda,equalchoose,order)
    else:  #若为多变量组合决策规则
        coe=BestFeature[1]
        #-------------划分结点文本nt-----------------
        nt=''
        for i in range(len(bf)):
            nt+='%+.3f'%coe[i]+FeatureName[bf[i]]
        nt+='<=%.3f'%-coe[-1]
        tree={nt:{}}
        #----------分支-----------
        m,n=Xt.shape
        X1=np.c_[Xt[:,bf],np.ones([m,1])]   #增加全1列
        left=np.dot(X1,coe)<=0  #左侧的样本
        Xt1=Xt[left,:];  Yt1=Yt[left]
        Xt2=Xt[~left,:]; Yt2=Yt[~left]
        
        if (len(Yt1)==0)|(len(Yt2)==0):
            tree=LabelName[most(Yt)]
        else:
            tree[nt]['是']=CreatTree(Xt1,Yt1,FeatureName,LabelName,considerfeature.copy(),DataType,rule,cut,equalcut,deep+1,normal,lamuda,equalchoose,order)
            tree[nt]['否']=CreatTree(Xt2,Yt2,FeatureName,LabelName,considerfeature.copy(),DataType,rule,cut,equalcut,deep+1,normal,lamuda,equalchoose,order)
    return tree