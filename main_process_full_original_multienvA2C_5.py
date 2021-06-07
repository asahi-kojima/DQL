import cupy as np
import tkinter as tk
import time
import math
import random
import copy
from collections import OrderedDict


from neural_net import neuralnet_function as nnf
from neural_net import neuralnet_class as nnc
from environment import block_env as benv

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym


def sign(x):
    if x>0:
        return 1.
    else:
        return -1.




GAMMA=0.995
NUM_EPISODES=100000

NUM_PROCESSES=32
NUM_ADVANCED_STEP = 5

learning_rate = 0.0001


max_grad_norm=0.5
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val,np.float32)
                self.v[key] = np.zeros_like(val,np.float32)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape):
        #基本情報
        self.observations = np.zeros((num_steps+1,num_processes,*obs_shape),np.float32)
        self.masks = np.ones((num_steps+1,num_processes, 1),np.float32)
        self.rewards = np.zeros((num_steps,num_processes, 1),np.float32)
        self.actions = np.zeros((num_steps,num_processes, 1),np.float32)

        self.returns = np.zeros((num_steps+1,num_processes, 1),np.float32)
        self.dout = np.zeros((num_steps+1,num_processes, 1),np.float32)
        self.index = 0

    def save(self, obs, action, reward, mask):
        self.observations[self.index + 1] = copy.copy(obs)
        self.masks[self.index + 1] = copy.copy(mask)
        self.rewards[self.index] = copy.copy(reward)
        self.actions[self.index] = copy.copy(action)

        self.index = (self.index + 1)%NUM_ADVANCED_STEP

    def after_update(self):
        self.observations[0] = copy.copy(self.observations[-1])

    def compute_returns(self, next_value):
        #next_valueのshapeは(num_processes,1)
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.shape[0])):
            self.returns[ad_step] = self.rewards[ad_step]+GAMMA*self.masks[ad_step + 1]*self.returns[ad_step + 1]

    def compute_dout(self):
        self.dout[-1] = 1
        for ad_step in reversed(range(self.rewards.shape[0])):
            self.returns[ad_step] = GAMMA*self.masks[ad_step + 1]*self.dout[ad_step + 1]





class Net():
    def __init__(self,params):
        weight_init_std=0.01
        self.params={}
        self.params['CW1']=params["arr_0"]#weight_init_std*np.random.randn(32,2,8,8)
        self.params['Cb1']=params["arr_1"]#np.ones(32)
        self.params['CW2']=params["arr_2"]#weight_init_std*np.random.randn(64,32,4,4)
        self.params['Cb2']=params["arr_3"]#np.ones(64)
        self.params['CW3']=params["arr_4"]#weight_init_std*np.random.randn(64,64,3,3)
        self.params['Cb3']=params["arr_5"]#np.ones(64)

        self.params['W1']=params["arr_6"]#weight_init_std*np.random.randn(5184,512)/np.sqrt(5184/2)
        self.params['b1']=params["arr_7"]#np.zeros(512)

        self.params['W_critic']=params["arr_8"]#weight_init_std*np.random.randn(512,1)/np.sqrt(512/2)
        self.params['b_critic']=params["arr_9"]#np.zeros(1)

        self.params['W_actor']=params["arr_10"]#weight_init_std*np.random.randn(512,3)/np.sqrt(512/2)
        self.params['b_actor']=params["arr_11"]#np.zeros(3)

        self.layers=OrderedDict()
        self.layers["Convolution1"]=nnc.Convolution(self.params['CW1'],self.params['Cb1'],stride=4)
        self.layers['Relu1']=nnc.Relu()
        self.layers["Convolution2"]=nnc.Convolution(self.params['CW2'],self.params['Cb2'],stride=2)
        self.layers['Relu2']=nnc.Relu()
        self.layers["Convolution3"]=nnc.Convolution(self.params['CW3'],self.params['Cb3'],stride=1)
        self.layers['Relu3']=nnc.Relu()
        self.layers["Connect"]=nnc.Connection()

        self.layers['Affine1']=nnc.Affine(self.params['W1'],self.params['b1'])

        self.layer_Affine_critic = nnc.Affine(self.params['W_critic'],self.params['b_critic'])
        self.layer_Affine_actor = nnc.Affine(self.params['W_actor'],self.params['b_actor'])




    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        critic_output = self.layer_Affine_critic.forward(x)
        actor_output = self.layer_Affine_actor.forward(x)

        return critic_output, actor_output

    def gradient(self, dout_critic, dout_actor):

        dout_critic = self.layer_Affine_critic.backward(dout_critic)
        dout_actor = self.layer_Affine_actor.backward(dout_actor)
        dout = dout_actor + dout_critic
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        self.grads={}
        self.grads['CW1']=self.layers['Convolution1'].dW
        self.grads['Cb1']=self.layers['Convolution1'].db
        self.grads['CW2']=self.layers['Convolution2'].dW
        self.grads['Cb2']=self.layers['Convolution2'].db
        self.grads['CW3']=self.layers['Convolution3'].dW
        self.grads['Cb3']=self.layers['Convolution3'].db

        self.grads['W1']=self.layers['Affine1'].dW
        self.grads['b1']=self.layers['Affine1'].db

        self.grads['W_critic']=self.layer_Affine_critic.dW
        self.grads['b_critic']=self.layer_Affine_critic.db

        self.grads['W_actor']=self.layer_Affine_actor.dW
        self.grads['b_actor']=self.layer_Affine_actor.db


    def get_action(self, x):
        #x=array(num_processes,*obs_shape)
        _, actor_output = self.forward(x)
        #出力値は(num_processes,3)
        #actor_output.size()はtorch.size([1,3]),actionとvalueは(1,1),torch.size([1,1])
        prob = nnf.softmax(actor_output)
        #print(prob)
        action =np.array([ np.random.choice(a=[0,1,2],size= 1, p = prob[i]) for i in range(x.shape[0])],np.float32)
        #actionは(NUM_PROCESSES,1)で出力
        return action.reshape(-1,1)


    def get_value(self,x):
        value, _ = self.forward(x)
        return value.reshape(-1,1)


    def evaluate_actions(self, x, actions):
        value, actor_output = self.forward(x)

        probs = nnf.softmax(actor_output) # (step*num_processes,3)
        #print("test1")
        #print(probs)
        action_probs = np.array( [probs[i][int(actions[i])] for i in range(actions.shape[0])] ,np.float32).reshape(-1,1) # (~,1)
        #print("test2")
        #print(action_probs)
        action_log_probs= np.log(action_probs+1e-7)# (~,1)
        self.probs = probs
        #print("test3")
        entropy = -np.sum(np.log(probs+1e-7)*probs)/probs.shape[0]
        #print("test4")
        return value, action_log_probs, entropy



class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.optimizer = Adam(lr = learning_rate)

    def update(self, rollouts):
        obs_shape = rollouts.observations.shape[2:]
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
        rollouts.observations[:-1].reshape(-1,2,100,100),
        rollouts.actions.reshape(-1,1) #(num_step*num_processes,1)
        )



        values = values.reshape(num_steps,num_processes, 1)
        action_log_probs = action_log_probs.reshape(num_steps,num_processes, 1)
        advantages = rollouts.returns[:-1]-values
        value_loss = np.mean(advantages**2)
        action_gain = np.mean(action_log_probs*advantages)
        total_loss = 0.5*value_loss-action_gain-0.01*entropy

        #逆伝搬の計算
        probs = self.actor_critic.probs #(step*process,3)

        tmp = np.zeros_like(probs,np.float32)
        act = rollouts.actions.reshape(-1,1)
        for i in range(num_steps*num_processes):
            tmp[i][int(act[i])]=(rollouts.returns[:-1].reshape(-1,1)-values.reshape(-1,1))[i][0]/(num_processes*num_steps)

        #dout_Jの計算
        dout_J = tmp*probs-(np.sum(tmp*probs,axis = -1).reshape(-1,1))*probs
        #dout_Entropyの計算
        entropy_sec = np.sum(probs*np.log(probs), axis =1).reshape(-1,1)
        dout_entropy = (probs*np.log(probs)-probs*entropy_sec)/(num_steps*num_processes)
        #dout_actorの計算
        dout_actor = -0.01*dout_entropy-dout_J


        rollouts.compute_dout()
        dout_value_loss = 2*advantages*(rollouts.dout[:-1]-1)/(num_steps*num_processes)
        dout_action_gain = action_log_probs*(rollouts.dout[:-1]-1)/(num_steps*num_processes)
        dout_critic = 0.5*dout_value_loss-dout_action_gain

        self.actor_critic.gradient(dout_critic = dout_critic.reshape(num_steps*num_processes,1),
                                            dout_actor = dout_actor.reshape(num_steps*num_processes,3))
        params = self.actor_critic.params
        grads = self.actor_critic.grads
        self.optimizer.update(params, grads)



class Environment:
    def __init__(self,save_flag = False, path = "./trained_params.npz"):
        self.save_flag = save_flag
        print("============ Load parameters ============")
        self.params = np.load(path)
        self.actor_critic = Net(self.params)
    def run(self):
        envs = [benv.Env() for i in range(NUM_PROCESSES)]
        n_in= (2,100,100)
        n_plot_in = (1,100,100)
        brain = Brain(self.actor_critic)

        #状態のサイズ
        obs_np = np.zeros([NUM_PROCESSES,*n_in],np.float32)
        obs_plot_np = np.zeros([NUM_PROCESSES,*n_plot_in],np.float32)

        #基本情報のリスト
        current_obs = np.zeros((NUM_PROCESSES,*n_in),np.float32)
        rollouts = RolloutStorage(NUM_ADVANCED_STEP,NUM_PROCESSES,n_in)
        episode_rewards = np.zeros([NUM_PROCESSES,1],np.float32)
        final_rewards = np.zeros([NUM_PROCESSES,1],np.float32)


        reward_np = np.zeros([NUM_PROCESSES,1],np.float32)
        done_np = np.zeros([NUM_PROCESSES,1],np.float32)
        each_step = [ 0 for _ in range(NUM_PROCESSES)]

        #状態の初期化
        obs = [env.reset() for env in envs]
        obs = np.array(obs,np.float32)
        current_obs = obs #(NUM_PROCESSES, 2, 100, 100)

        full_total_ref=[ 0 for _ in range(NUM_PROCESSES)]
        total_ref=[ 0 for _ in range(NUM_PROCESSES)]
        noop = np.random.randint(0,30,(NUM_PROCESSES,1)).astype(np.float32)

        elapsed_episode = [ 0 for _ in range(NUM_PROCESSES)]

        life = [ 0 for _ in range(NUM_PROCESSES)]

        for j in range(NUM_EPISODES):
            for step in range(NUM_ADVANCED_STEP):
                #actionの取得
                #with torch.no_grad():
                actions = self.actor_critic.get_action(rollouts.observations[step]) #(NUM_PROCESSES,1)のベクトル
                #No-operationの設定
                for i in range(NUM_PROCESSES):
                    if each_step[i]<=noop[i]:
                        actions[i] = np.array([1],np.float32)
                    #メインプロセスのステップ実行
                    #print(envs[i].step(2*(actions.tolist()[i][0]-1)/envs[i].scale)[0].shape)
                    #print(obs_np[i].shape)
                    obs_np[i], done_np[i],_= envs[i].step(2*(actions.tolist()[i][0]-1)/envs[i].scale)
                    total_ref[i] += envs[i].ref_n



                    #ボールが地面に着いた場合
                    if done_np[i]:
                    #描画設定箇所
                        life[i] += 1
                        full_total_ref[i]+=total_ref[i]
                        if i == 0:
                            print(f'{elapsed_episode[i]:04} | {int(each_step[i]):04} | {len(envs[i].lst):02} |',\
                             f'{total_ref[i]:02} | {int(10*full_total_ref[i]/(elapsed_episode[i]+1))/10} |',\
                              f'{int(sum([10*full_total_ref[i]/(elapsed_episode[i]+1) for i in range(NUM_PROCESSES)])/NUM_PROCESSES)/10} |',\
                               f'{int(min(elapsed_episode))} ')
                            print([int(10*full_total_ref[i]/(elapsed_episode[i]+1))/10 for i in range(NUM_PROCESSES)])
                            print("=============================================================================")
                        if total_ref[i]==0:
                            reward_np[i] = np.array([-1.],np.float32)
                        else:
                            reward_np[i] = np.array([1-len(envs[i].lst)/envs[i].block_num],np.float32)

                        each_step[i] = 0
                        if life[i] <= 3:
                            obs_np[i] = envs[i].reset_tmp(envs[i].lst,envs[i].tmp)
                        else:
                            life[i]=0
                            obs_np[i] = envs[i].reset()
                        noop[i] = random.randint(0,30)
                        total_ref[i]=0
                        elapsed_episode[i]+=1

                    #問題なくステップが進んだ場合
                    else:
                        if envs[i].mouse_x < 10 or 90 < envs[i].mouse_x:
                            if envs[i].ref_n==0:
                                reward_np[i] = -0.01
                            else:
                                reward_np[i] = 1./NUM_PROCESSES
                        else:
                            if envs[i].ref_n==0:
                                reward_np[i] = 0.
                            else:
                                reward_np[i] = 1./NUM_PROCESSES
                        each_step[i] += 1

                #更新の準備
                reward = reward_np
                episode_rewards += reward

                masks = np.array([[0.] if done_ else [1.] for done_ in done_np],np.float32)
                final_rewards *= masks
                final_rewards += (1-masks)*episode_rewards
                episode_rewards *= masks

                obs = obs_np
                current_obs = obs
                #全てTensor
                rollouts.save(current_obs, actions, reward, masks)
            #Advanced-step終了後の更新
            #with torch.no_grad():
            next_value = self.actor_critic.get_value(rollouts.observations[-1])
            rollouts.compute_returns(next_value)
            brain.update(rollouts)
            rollouts.after_update() #更新


            if np.average(elapsed_episode) >= 100+1:
                print("Finish : ",int(sum([10*full_total_ref[i]/(elapsed_episode[i]+1) for i in range(NUM_PROCESSES)])/NUM_PROCESSES)/10)
                break
        #test
        return int(sum([10*full_total_ref[i]/(elapsed_episode[i]+1) for i in range(NUM_PROCESSES)])/NUM_PROCESSES)/10






full_results = []
results = []
tmp_max=0
with open("./model_best.txt","r") as f:
    tmp_max = float(f.read())

for seq in range(100):
    path = "./trained_params.npz"
    print("number [",seq, "] loop started")

    for loop in range(1000):
        flag = False
        print("Now is ",seq,"-",loop,"time's loop : current best score is ",tmp_max)
        env=Environment(save_flag = flag, path = path)
        ave = env.run()
        results.append(ave)

        if tmp_max <= ave:
            print("maximum value of average is updated : ",ave)
            tmp_max = ave
            print("+++++++++++ save parameters +++++++++++++")
            np.savez("./trained_params",
                        env.actor_critic.params["CW1"].astype(np.float32),
                        env.actor_critic.params["Cb1"].astype(np.float32),
                        env.actor_critic.params["CW2"].astype(np.float32),
                        env.actor_critic.params["Cb2"].astype(np.float32),
                        env.actor_critic.params["CW3"].astype(np.float32),
                        env.actor_critic.params["Cb3"].astype(np.float32),
                        env.actor_critic.params["W1"].astype(np.float32),
                        env.actor_critic.params["b1"].astype(np.float32),
                        env.actor_critic.params["W_critic"].astype(np.float32),
                        env.actor_critic.params["b_critic"].astype(np.float32),
                        env.actor_critic.params["W_actor"].astype(np.float32),
                        env.actor_critic.params["b_actor"].astype(np.float32))
            with open("./model_best.txt","w") as f:
                f.write(str(f'{tmp_max}'))

        else:
            print("maximum value of average is not updated : ",tmp_max)

        path = "./trained_params.npz"
        del env
    full_results.append(sum(results)/len(results))
