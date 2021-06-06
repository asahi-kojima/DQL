import cupy as np


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






class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class BatchN:
    def __init__(self,g,b):
        self.g=g
        self.b=b
        self.mu=None
        self.sigma=None


    def forward(self,x):
        self.k=1e-7
        mu=np.sum(x,axis=1)/(x.shape[1])
        self.mu=mu.reshape(x.shape[0],1)

        sigma2=np.sum((x-self.mu)**2,axis=1)/(x.shape[1])
        self.sigma2=sigma2.reshape(len(sigma2),1)

        self.xp=(x-self.mu)/np.sqrt(self.sigma2+self.k)
        self.y=self.g*self.xp+self.b
        return self.y

    def backward(self,dout):
        self.dg=np.sum(self.xp*dout)
        self.db=np.sum(dout,axis=0)

        self.grads=np.empty((0,dout.shape[1]), int)
        for i in range(dout.shape[0]):
            m=(self.g/np.sqrt(self.sigma2[i]+self.k))*np.dot(dout[i],np.identity(dout.shape[1])-(np.ones(dout.shape[1])+np.outer(self.y[i],self.y[i]))/(dout.shape[1]))
            self.grads=np.append(self.grads,m.reshape(1,len(m)),axis=0)
        return self.grads

class BatchNew:
    def __init__(self):
        self.mu=None
        self.sigma=None


    def forward(self,x):
        self.k=1e-7
        mu=np.sum(x,axis=1)/(x.shape[1])
        self.mu=mu.reshape(x.shape[0],1)

        sigma2=np.sum((x-self.mu)**2,axis=1)/(x.shape[1])
        self.sigma2=sigma2.reshape(len(sigma2),1)

        self.xp=(x-self.mu)/np.sqrt(self.sigma2+self.k)
        self.y=self.xp
        return self.y

    def backward(self,dout):

        self.grads=np.empty((0,dout.shape[1]), int)
        for i in range(dout.shape[0]):
            m=(1/np.sqrt(self.sigma2[i]+self.k))*np.dot(dout[i],np.identity(dout.shape[1])-(np.ones(dout.shape[1])+np.outer(self.y[i],self.y[i]))/(dout.shape[1]))
            self.grads=np.append(self.grads,m.reshape(1,len(m)),axis=0)
        return self.grads


class Connection:
    def __init__(self):
        pass

    def forward(self,x):
        self.N,self.C,self.H,self.W=x.shape
        y=x.reshape(self.N,-1)
        return y


    def backward(self,dout):
        out=dout
        out=out.reshape(self.N,self.C,self.H,self.W)
        return out


class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b#(1,FN)のベクトル
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)

        col=myim2col(x,FH,FW,self.stride,self.pad)
        col_w=self.W.reshape(FN,-1).T
        out=np.dot(col,col_w)+self.b

        self.x = x
        self.col=col
        self.col_W = col_w

        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        return out

    def backward(self,dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = mycol2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        self.N,self.C,H,W=x.shape
        self.sheip=x.shape
        self.out_h=int(1+(H+2*self.pad-self.pool_h)/self.stride)
        self.out_w=int(1+(W+2*self.pad-self.pool_w)/self.stride)
        col=myim2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col=col.reshape(-1,self.pool_h*self.pool_w)
        out=np.max(col,axis=1)
        self.arg_max = np.argmax(col, axis=1)
        out=out.reshape(self.N,self.C,self.out_h,self.out_w)

        return out

    def backward(self,dout):
        out=dout
        out=out.reshape(1,out.size)
        tmp=np.zeros((out.size,self.pool_h*self.pool_w))
        tmp[np.arange(out.size),self.arg_max]=out
        dx=mycol2im(tmp,self.sheip,self.pool_h,self.pool_w,stride=self.stride,pad=self.pad)

        return dx














class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
