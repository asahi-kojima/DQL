import tkinter as tk
import time
import math

root=tk.Tk()
root.resizable(False,False)
def sign(x):
    if x>0:
        return 1.
    else:
        return -1.

#========================ハイパーパラメーター========================
scale=0.8
DT=0.01
block_vertical_num=3
block_horizontal_num=10
#========================台に関する設定========================
width=800*scale
height=800*scale
height_b=100*scale
cvs=tk.Canvas(root,width=width,height=height,bg="black")
cvs.pack()
cvs.create_line(0,
                height-height_b,
                width,
                height-height_b,
                fill="white")



#========================ボールに関する変数========================
ball_radius=10*scale
size=((width/block_horizontal_num)//2)
state_ball={"x":(width/2), "y":2*size*(block_vertical_num+3),
            "vx":-700.*scale, "vy":-700.*scale}


#========================ブロックの設定========================
blocks={}
block_num=int(block_vertical_num*block_horizontal_num)
size=((width/block_horizontal_num)//2)

lst=[chr(j+97)+chr(i+97) for j in range(block_vertical_num) for i in range(block_horizontal_num)]
tmp=[chr(j+97)+chr(i+97) for j in range(block_vertical_num) for i in range(block_horizontal_num)]
lab=[]
class Block:
    def __init__(self,x,y,size,tag,color="white"):
        self.center_x=x
        self.center_y=y
        self.left=x-size
        self.right=x+size
        self.top=y-size
        self.bottom=y+size
        self.size=size
        self.tag=tag
        self.color=color

    def plot(self):
        cvs.create_rectangle(self.left,
                            self.top,
                            self.right,
                            self.bottom,
                            fill=self.color,
                            tag=self.tag)

    def delete(self):
        global tmp
        cvs.create_rectangle(self.left,
                            self.top,
                            self.right,
                            self.bottom,
                            fill="white",tag=self.tag)
        cvs.delete(self.tag)
        tmp.remove(self.tag)

    def judge_collision_block(self,x,y,vx,vy):
        x_next=x+vx*DT
        y_next=y+vy*DT
        if abs(x-self.left)<1.2*ball_radius and self.top<y<self.bottom:
            self.delete()
            return {"x":x, "y":y, "vx":-vx, "vy":vy}
        if abs(x-self.right)<1.2*ball_radius and self.top<y<self.bottom:
            self.delete()
            return {"x":x, "y":y, "vx":-vx, "vy":vy}
        if abs(y-self.top)<1.2*ball_radius and self.left<x<self.right:
            self.delete()
            return {"x":x, "y":y, "vx":vx, "vy":-vy}
        if abs(y-self.bottom)<1.2*ball_radius and self.left<x<self.right:
            self.delete()
            return {"x":x, "y":y, "vx":vx, "vy":-vy}
        return {"x":x, "y":y, "vx":vx, "vy":vy}



for j in range(block_vertical_num):
    for i in range(block_horizontal_num):
        blocks[chr(j+97)+chr(i+97)]=Block(size+2*size*i,
                                        size+2*size*j,size=size,
                                    tag=chr(j+97)+chr(i+97),
                                    color=f'#ff0{2*i}{2*i}{2*i}')
        blocks[chr(j+97)+chr(i+97)].plot()


#========================マウス操作に関する変数========================
mouse_x=0
mouse_y=height-height_b
bar_width=70*scale
bar_height=10*scale



#========================衝突判定========================
def judge_collision_wall(x,y,vx,vy):
    x_next=x+vx*DT
    y_next=y+vy*DT
    vx_next=vx
    vy_next=vy
    if x_next-ball_radius<0:
        return {"x":x_next, "y":y_next, "vx":-vx_next, "vy":vy_next}
    if x_next+ball_radius>width:
        return {"x":x_next, "y":y_next, "vx":-vx_next, "vy":vy_next}
    if y_next-ball_radius<0:
        return {"x":x_next, "y":y_next, "vx":vx_next, "vy":-vy_next}
    if y_next+ball_radius>height:
        return {"x":x_next, "y":y_next, "vx":vx_next, "vy":-vy_next}
    return {"x":x_next, "y":y_next, "vx":vx_next, "vy":vy_next}


def judge_collision_bar(x,y,vx,vy):
    x_next=x+vx*DT
    y_next=y+vy*DT
    vx_next=vx
    vy_next=vy
    if abs(y-(mouse_y-bar_height))<1.2*ball_radius and mouse_x-bar_width<x<mouse_x+bar_width and vy>0:
        velo=math.sqrt(vx**2+vy**2)
        theta=(x-mouse_x)*(math.pi/4)/bar_width
        return {"x":x, "y":y, "vx":velo*math.sin(theta), "vy":-velo*math.cos(theta)}
    return {"x":x, "y":y, "vx":vx, "vy":vy}


#========================マウスの動作========================
def mouse_move(e):
    global mouse_x
    mouse_x=e.x
    if mouse_x-bar_width<0:
        mouse_x=bar_width
    elif mouse_x+bar_width>width:
        mouse_x=width-bar_width
    else:
        mouse_x=mouse_x

def draw_txt(txt, x, y, siz, col, tg):
    fnt = ("Times New Roman", siz, "bold")
    cvs.create_text(x+2, y+2, text=txt, fill="black", font=fnt, tag=tg)
    cvs.create_text(x, y, text=txt, fill=col, font=fnt, tag=tg)


#========================ゲームのメイン処理========================
def main_process():
    global DT,state_ball,lst
    #ボールとバーの描画
    cvs.delete("ball")
    cvs.create_oval(state_ball["x"]-ball_radius,state_ball["y"]-ball_radius,
                    state_ball["x"]+ball_radius,state_ball["y"]+ball_radius,
                    fill="yellow",tag="ball")
    cvs.delete("bar")
    cvs.create_rectangle(mouse_x+bar_width,
                        mouse_y+bar_height,
                        mouse_x-bar_width,
                        mouse_y-bar_height,
                        fill="cyan",
                        width=0,
                        tag="bar")
    """
    cvs.update()
    cvs.postscript(file="hogehoge.pdf", colormode='color')
    """
    #ボールの衝突判定
    state_ball=judge_collision_wall(*list(state_ball.values()))
    state_ball=judge_collision_bar(*list(state_ball.values()))
    #ブロックとの衝突判定
    if state_ball["y"]<(block_vertical_num+1)*2*size:
        for i in lst:
            state_ball=blocks[i].judge_collision_block(*list(state_ball.values()))
    lst=tmp.copy()
    #cvs.delete("OVER")
    #if abs(state_ball["y"]-height+bar_height)<1.1*ball_radius:
    #    cvs.create_rectangle((300-140)*scale,(600-30)*scale,(300+140)*scale,(600+30)*scale,fill="orange",width=0,tag="OVER")
    #    draw_txt("GAME OVER",300*scale,600*scale,int(40*scale),"white","OVER")
    root.after(10,main_process)

#========================稼働準備========================
root.bind("<Motion>",mouse_move)
main_process()
root.mainloop()
