# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:03:52 2019

@author: Soly Liang
"""

import matplotlib.pyplot as plt
import numpy as np

#设置出图显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#由西瓜数据集3.0
Tree_test={'纹理=?': 
         {'清晰': 
             {'根蒂=?':
                 {'蜷缩': '好瓜',
                  '稍蜷':
                      {'色泽=?': 
                          {'青绿': '好瓜',
                           '乌黑': 
                               {'触感=?': 
                                   {'硬滑': '好瓜',
                                    '软粘': '坏瓜'}}, 
                           '浅白': '好瓜','aaa':111,'bbb':222}}, 
                  '硬挺': '坏瓜'}},
          '稍糊': 
              {'触感=?':
                  {'硬滑': {'A':{'a1':{'B':{'b1':1,'b2':2}},'a2':2}}, 
                   '软粘': '好瓜'}}, 
          '模糊': '坏瓜'}}

def Transform(Tree,struct=None,RootChain=None,deep=0,mark=None):
    # 将字典数据结构的决策树进行转换
    if struct==None:
        struct=[[]]
    if RootChain==None:
        RootChain=[]
    RootFeature=list(Tree.keys())[0]
    if len(struct)<=deep:
        struct.append([])
    struct[deep].append([RootChain,RootFeature,mark,False])
    subs=Tree[RootFeature]
    n=len(subs)
    x0=-(n-1)/2
    for key in subs:
        if type(subs[key])==dict:
            Transform(subs[key],struct,RootChain+[x0],deep+1,key)
        else:
            if len(struct)-1<deep+1:
                struct.append([])
            struct[deep+1].append([RootChain+[x0],subs[key],key,True])
        x0+=1
    return struct


def Cal_d(struct):
    # 计算每一代的基础间距(不包括第0代),存于列表d中
    deep=len(struct)
    d=[1]*deep  #最后一个多余的,但是方便于编程
    mingap=0.5
    for generation in range(-1,-(deep-1),-1): #从最下代往上至第第三代
        chain=np.array([nodes[0] for nodes in struct[generation]])
        for i in range(2,len(chain)):
            if (chain[i,:-1]-chain[i-1,:-1]).any(): #当同一代相邻两个结点非亲兄弟时
                if np.dot(chain[i,:]-chain[i-1,:],d[:generation])<mingap:  #当间距较小时
                    FirstSplit=list((chain[i,:]-chain[i-1,:])!=0).index(True)  #哪一代开始不同祖先
                    #更新这一代的基础间距↓↓↓
                    d[FirstSplit]=(mingap-np.dot(chain[i,FirstSplit+1:]\
                                  -chain[i-1,FirstSplit+1:],d[FirstSplit+1:generation]))\
                                  /(chain[i,FirstSplit]-chain[i-1,FirstSplit])
    return d[:-1]

def Cal_x(node,d):
    # 计算每个结点的横坐标
    chain=node[0]
    if len(chain)==0:
        return 0
    return np.dot(chain,d[:len(chain)])

def draw(tree,title='',xzoom=1,yzoom=1):
    # 绘制决策树
    # 输入 tree为决策树，数据结构为字典型，xzoom,yzoom为x，y坐标的缩放系数，默认为1
    plt.figure()
    plt.title(title)
    if type(tree)!=dict:
        plt.annotate(str(tree),xy=[0.5,0.5],bbox={'boxstyle':'circle','fc':'1'},ha='center')
        plt.axis([0,1,0,1])
        plt.axis('off')
        plt.show()
        return
    struct=Transform(tree)   #首先将决策树数据结构进行转换
    d=Cal_d(struct)
    deep=len(struct)
    arrow={'arrowstyle':'-'}   #不用箭头，用线段连接
    leafbox={'boxstyle':"round4",'fc':'1'}    #叶结点为圆
    noleafbox={'boxstyle':"square",'fc':'1'}  #非叶结点为矩形框
    plt.annotate(struct[0][0][1],xy=[0,0],bbox=noleafbox,ha="center")
    xmin=0
    xmax=0
    for g in range(1,deep):
       for node in struct[g]:
            xytext=[xzoom*Cal_x(node,d),-g*yzoom]
            father=[n[0] for n in struct[g-1]].index(node[0][:-1])
            xy=[xzoom*Cal_x(struct[g-1][father],d),-(g-1)*yzoom-yzoom*0.1]
            if node[3]:
                plt.annotate(node[1],xy=xy,xytext=xytext,bbox=leafbox,arrowprops=arrow,ha="center")
            else:
                plt.annotate(node[1],xy=xy,xytext=xytext,bbox=noleafbox,arrowprops=arrow,ha="center")
            linex=(xytext[0]+xy[0])/2+node[0][-1]*0.1*xzoom
            liney=(xytext[1]+xy[1])/2
            if node[0][-1]>=0:
                plt.text(linex,liney,node[2],ha="left",va="center")
            else:
                plt.text(linex,liney,node[2],ha="right",va="center")
            xmax=max([xmax,xytext[0]])
            xmin=min([xmin,xytext[0]])
    
    plt.axis([xmin-xzoom*0.1,xmax+xzoom*0.1,-(deep-1)*yzoom,yzoom*0.5])
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 测试效果
    draw('我是一片叶','只有一个叶结点的树')
    draw(Tree_test,'测试绘制一颗决策树')