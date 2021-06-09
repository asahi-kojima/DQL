import cupy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフローへの対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(x,t):
    if x.ndim==1:
        t=t.reshape(1,t.size)
        x=x.reshape(1,x.size)
    delta=1e-7
    return -np.sum(t*np.log(x+delta))/(x.shape[0])


#numpyとcupyで作動が異なるので注意
def myim2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col=np.zeros((N,out_h,out_w,C*filter_h*filter_w))
    for i in range(out_h):
        for j in range(out_w):
            col[:,i,j,:]=img[:,:,i*stride:i*stride+filter_h,j*stride:j*stride+filter_w].reshape(N,-1)
    col=col.reshape(N*out_h*out_w,-1)
    return col




def mycol2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    input_data=np.zeros(input_shape)
    tmp_input_data=np.zeros((N,C,H+2*pad,W+2*pad))
    col=col.reshape(N,out_h*out_w,-1)

    for i in range(out_h):
        for j in range(out_w):
            tmp_input_data[:,:,i*stride:i*stride+filter_h,j*stride:j*stride+filter_w]=\
            tmp_input_data[:,:,i*stride:i*stride+filter_h,j*stride:j*stride+filter_w]+col[:,i*out_w+j,:].reshape(N,-1,filter_h,filter_w)
    input_data[:,:,:,:]=tmp_input_data[:,:,pad:pad+H,pad:pad+W]
    return input_data









def numerical_gradient1(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    for i in range(len(x)):
        tmp_val=x[i]
        x[i]=tmp_val+h
        fxh1=f(x)
        x[i]=tmp_val-h
        fxh2=f(x)
        grad[i]=(fxh1-fxh2)/(2*h)
        x[i]=tmp_val
    return grad

def numerical_gradient2(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp_val=x[i,j]
            x[i,j]=tmp_val+h
            fxh1=f(x)
            x[i,j]=tmp_val-h
            fxh2=f(x)
            grad[i,j]=(fxh1-fxh2)/(2*h)
            x[i,j]=tmp_val
    return grad


def numerical_gradient(f,x):
    if x.ndim==1:return numerical_gradient1(f,x)
    if x.ndim==2:return numerical_gradient2(f,x)
#以下は試行錯誤中：エラー要因が不明
