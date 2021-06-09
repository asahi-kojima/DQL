import cupy as np
import tkinter as tk
import tkinter as tk
import time
import math
import random
import copy
from collections import OrderedDict



class Block:
    def __init__(self,x,y,size,tag,DT,ball_radius,color="white"):
        self.center_x=x
        self.center_y=y
        self.left=x-size
        self.right=x+size
        self.top=y-size
        self.bottom=y+size
        self.size=size
        self.tag=tag
        self.color=color
        self.DT=DT
        self.ball_radius = ball_radius
    def delete(self,lst):
        lst.remove(self.tag)

    def judge_collision_block(self,x,y,vx,vy,lst):
        x_next=x+vx*self.DT
        y_next=y+vy*self.DT
        if abs(x-self.left)<1.2*self.ball_radius and self.top<y<self.bottom:
            self.delete(lst)
            return {"x":x, "y":y, "vx":-vx, "vy":vy}, self.tag
        if abs(x-self.right)<1.2*self.ball_radius and self.top<y<self.bottom:
            self.delete(lst)
            return {"x":x, "y":y, "vx":-vx, "vy":vy}, self.tag
        if abs(y-self.top)<1.2*self.ball_radius and self.left<x<self.right:
            self.delete(lst)
            return {"x":x, "y":y, "vx":vx, "vy":-vy}, self.tag
        if abs(y-self.bottom)<1.2*self.ball_radius and self.left<x<self.right:
            self.delete(lst)
            return {"x":x, "y":y, "vx":vx, "vy":-vy}, self.tag
        return {"x":x, "y":y, "vx":vx, "vy":vy},None

class Env:
    def __init__(self):
        self.obs = 0
        self.obs_plot = 0
        self.reward = 0
        self.done = None


        self.scale = 1/8
        self.DT = 0.01
        self.block_vertical_num = 3
        self.block_horizontal_num = 10

        self.width = 800*self.scale
        self.height = 800*self.scale
        self.height_b = 100*self.scale

        self.ball_radius = 10*self.scale

        self.blocks = {}
        self.block_num = int(self.block_vertical_num*self.block_horizontal_num)
        self.size = ((self.width/self.block_horizontal_num)//2)

        degr = (random.random()-0.5)*math.pi*2/3
        velo = 700*math.sqrt(2)
        self.state_ball = {"x":self.width*random.random(), "y":2*self.size*(self.block_vertical_num+3),
                            "vx":-velo*self.scale*math.sin(degr),
                            "vy":-velo*self.scale*math.cos(degr)}

        self.lst = [(j,i) for j in range(self.block_vertical_num) for i in range(self.block_horizontal_num)]
        self.tmp = [(j,i) for j in range(self.block_vertical_num) for i in range(self.block_horizontal_num)]

        for j in range(self.block_vertical_num):
            for i in range(self.block_horizontal_num):
                self.blocks[(j,i)] = Block(self.size+2*self.size*i,
                                            self.size+2*self.size*j,size=self.size,
                                            tag=(j,i),DT = self.DT,ball_radius=self.ball_radius,
                                            color=f'#ff0{2*i}{2*i}{2*i}')

        self.mouse_x = self.width/2
        self.mouse_y = self.height-self.height_b
        self.bar_width = 70*self.scale
        self.bar_height = 10*self.scale

        self.ref_n = 0
        self.flag = False

        self.base_pic = np.zeros((1,2,100,100),np.float32)
        for n in self.lst:
            j,i = n
            self.base_pic[0,:,int(2*self.size*j*(100/self.width)):int(2*self.size*(j+1)*(100/self.width)),\
                                int(2*self.size*i*(100/self.width)):int(2*self.size*(i+1)*(100/self.width))]=0.01
        self.deleted_tag = None

    def reset(self):
        self.__init__()
        self.obs = self.create_pic(
        self.lst,
        self.tmp,
        self.state_ball["x"],
        self.state_ball["y"],
        self.mouse_x,
        self.mouse_y,
        self.state_ball["x"]+self.DT*self.state_ball["vx"],
        self.state_ball["y"]+self.DT*self.state_ball["vy"],
        self.size,None)

        return self.obs.reshape(2,100,100)

    def reset_tmp(self,lst_pre,tmp_pre):
        self.__init__()
        self.obs = self.create_pic(
        lst_pre,
        tmp_pre,
        self.state_ball["x"],
        self.state_ball["y"],
        self.mouse_x,
        self.mouse_y,
        self.state_ball["x"]+self.DT*self.state_ball["vx"],
        self.state_ball["y"]+self.DT*self.state_ball["vy"],
        self.size,deleted_tag = None, tmp_flag = True)
        self.lst = copy.copy(lst_pre)
        self.tmp = copy.copy(tmp_pre)
        return self.obs.reshape(2,100,100)

    def judge_collision_bar(self,x,y,vx,vy):
        x_next=x+vx*self.DT
        y_next=y+vy*self.DT
        vx_next=vx
        vy_next=vy
        if abs(y-(self.mouse_y-self.bar_height))<1.2*self.ball_radius and self.mouse_x-self.bar_width<x<self.mouse_x+self.bar_width and vy>0:
            velo=math.sqrt(vx**2+vy**2)
            theta=(x-self.mouse_x)*(math.pi/4)/self.bar_width
            self.ref_n=1
            return {"x":x, "y":y, "vx":velo*math.sin(theta), "vy":-velo*math.cos(theta)}
        self.ref_n=0
        return {"x":x, "y":y, "vx":vx, "vy":vy}


    def judge_collision_wall(self,x,y,vx,vy):
        x_next=x+vx*self.DT
        y_next=y+vy*self.DT
        vx_next=vx
        vy_next=vy
        if x_next-self.ball_radius<0 and vx < 0:
            return {"x":x_next, "y":y_next, "vx":-vx_next, "vy":vy_next}
        if x_next+self.ball_radius>self.width and vx>0:
            return {"x":x_next, "y":y_next, "vx":-vx_next, "vy":vy_next}
        if y_next-self.ball_radius<0 and vy<0:
            return {"x":x_next, "y":y_next, "vx":vx_next, "vy":-vy_next}
        if y_next+self.ball_radius>self.height and vy>0:
            return {"x":x_next, "y":y_next, "vx":vx_next, "vy":-vy_next}
        return {"x":x_next, "y":y_next, "vx":vx_next, "vy":vy_next}

    def create_pic(self,lst,lst2,x,y,x_b,y_b,X,Y,size,deleted_tag,tmp_flag = False):
        if tmp_flag == False:
            pic=np.zeros((1,2,100,100),np.float32)
            pic[:,0,:,:]+=self.base_pic[:,0,:,:]
            if deleted_tag != None:
                j,i = deleted_tag
                self.base_pic[:,:,int(2*size*j*(100/self.width)):int(2*size*(j+1)*(100/self.width))\
                    ,int(2*size*i*(100/self.width)):int(2*size*(i+1)*(100/self.width))]=0




            pic[0,0,min(max(int(y),0),99),max(int(x)-2,0):min(int(x)+2+1,99)]=1.0
            pic[0,0,min(max(int(y)-1,0),99),max(int(x)-1,0):min(int(x)+1+1,99)]=0.9
            pic[0,0,min(max(int(y)-2,0),99),min(max(int(x),0),99)]=0.8
            pic[0,0,min(max(int(y)+1,0),99),max(int(x)-1,0):min(int(x)+1+1,99)]=0.9
            pic[0,0,min(max(int(y)+2,0),99),min(max(int(x),0),99)]=0.8



            pic[0,1,min(max(int(Y),0),99),max(int(X)-2,0):min(int(X)+2+1,99)]=1.0
            pic[0,1,min(max(int(Y)-1,0),99),max(int(X)-1,0):min(int(X)+1+1,99)]=0.9
            pic[0,1,min(max(int(Y)-2,0),99),min(max(int(X),0),99)]=0.8
            pic[0,1,min(max(int(Y)+1,0),99),max(int(X)-1,0):min(int(X)+1+1,99)]=0.9
            pic[0,1,min(max(int(Y)+2,0),99),min(max(int(X),0),99)]=0.8

            pic[0,0,int(y_b),max(int(x_b-self.bar_width),0):min(int(x_b+self.bar_width),99)]=1
            pic[0,1,int(y_b),max(int(x_b-self.bar_width),0):min(int(x_b+self.bar_width),99)]=1

            pic[:,1,:,:]+=self.base_pic[:,1,:,:]

            return pic

        else:
            pic=np.zeros((1,2,100,100),np.float32)
            lst = [i for i in [(j,i) for j in range(self.block_vertical_num) for i in range(self.block_horizontal_num)] if i not in lst]
            lst2 = [i for i in [(j,i) for j in range(self.block_vertical_num) for i in range(self.block_horizontal_num)] if i not in lst2]
            for deleted_tag in lst:
                j,i = deleted_tag
                self.base_pic[0,0,int(2*size*j*(100/self.width)):int(2*size*(j+1)*(100/self.width))\
                    ,int(2*size*i*(100/self.width)):int(2*size*(i+1)*(100/self.width))]=0
            for deleted_tag in lst2:
                j,i = deleted_tag
                self.base_pic[0,1,int(2*size*j*(100/self.width)):int(2*size*(j+1)*(100/self.width))\
                    ,int(2*size*i*(100/self.width)):int(2*size*(i+1)*(100/self.width))]=0


            pic[0,0,min(max(int(y),0),99),max(int(x)-2,0):min(int(x)+2+1,99)]=1.0
            pic[0,0,min(max(int(y)-1,0),99),max(int(x)-1,0):min(int(x)+1+1,99)]=0.9
            pic[0,0,min(max(int(y)-2,0),99),min(max(int(x),0),99)]=0.8
            pic[0,0,min(max(int(y)+1,0),99),max(int(x)-1,0):min(int(x)+1+1,99)]=0.9
            pic[0,0,min(max(int(y)+2,0),99),min(max(int(x),0),99)]=0.8



            pic[0,1,min(max(int(Y),0),99),max(int(X)-2,0):min(int(X)+2+1,99)]=1.0
            pic[0,1,min(max(int(Y)-1,0),99),max(int(X)-1,0):min(int(X)+1+1,99)]=0.9
            pic[0,1,min(max(int(Y)-2,0),99),min(max(int(X),0),99)]=0.8
            pic[0,1,min(max(int(Y)+1,0),99),max(int(X)-1,0):min(int(X)+1+1,99)]=0.9
            pic[0,1,min(max(int(Y)+2,0),99),min(max(int(X),0),99)]=0.8

            pic[0,0,int(y_b),max(int(x_b-self.bar_width),0):min(int(x_b+self.bar_width),99)]=1
            pic[0,1,int(y_b),max(int(x_b-self.bar_width),0):min(int(x_b+self.bar_width),99)]=1

            pic_total = pic + self.base_pic

            return pic_total


    def step(self,mouse_diff):
        mouse_x_tmp=self.mouse_x+mouse_diff*self.scale
        if 0<mouse_x_tmp<self.width:
            self.mouse_x=mouse_x_tmp
        else:
            self.mouse_x=self.mouse_x
        X=self.state_ball["x"]
        Y=self.state_ball["y"]
        self.state_ball=self.judge_collision_wall(*list(self.state_ball.values()))
        self.state_ball=self.judge_collision_bar(*list(self.state_ball.values()))
        #ブロックとの衝突判定
        if self.state_ball["y"]<(self.block_vertical_num+1)*2*self.size:
            for i in self.lst:
                self.state_ball, self.deleted_tag=self.blocks[i].judge_collision_block(*list(self.state_ball.values()),self.tmp)
                if self.deleted_tag != None:
                    break

        if self.state_ball["y"]>97 or len(self.tmp)==0:
            self.flag=True


        obs=self.create_pic(self.lst,self.tmp,X,Y,self.mouse_x,self.mouse_y,\
                            self.state_ball["x"],self.state_ball["y"],size=self.size,deleted_tag=self.deleted_tag)
        self.lst=self.tmp.copy()
        self.deleted_tag = None
        return obs.reshape(2,100,100),self.flag, 0
