import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import gym
import time

# 超参数
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


### DDPG ###
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2+a_dim+1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim],'s')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor') # actor网络的权重
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay = 1-TAU) # 滑动平均，控制模型更新的速度，decay越大越趋于稳定 shadow_variable=decay*shadow_variable+(1−decay)*variable

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        
        target_update = [ema.apply(a_params), ema.apply(c_params)] # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter) # 替换target网络参数
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)  
        
        a_loss = -tf.reduce_mean(q)  # 让q最大
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params) # actor网络训练

        with tf.control_dependencies(target_update): # 这里soft replacement
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params) # critic网络训练
        
        self.sess.run(tf.global_variables_initializer())
        # 初始化部分结束
    
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0] #self.a = self._build_a(self.S,)
    
    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :] # total
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:] 
        # 这段代码的主要目的是从记忆池中随机选择一批经验，将这些经验分成状态、动作、奖励和下一个状态，
        # 并用于训练深度强化学习模型，如神经网络。这种经验回放的机制有助于减少数据的相关性，
        # 提高训练的稳定性，并能更有效地利用之前的经验来训练模型。
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False # 没写默认true
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
        
    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s)+tf.matmul(a, w1_a)+b1)
            return tf.layers.dense(net, 1, trainable=trainable) #Q(s,a)
        


### training ###
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0] # 有多少种状态
a_dim = env.action_space.shape[0] # 有多少种动作 离散动作空间没有shape
a_bound = env.action_space.high # 动作上限 最大值

ddpg = DDPG(a_dim, s_dim, a_bound)
var = 3 # 控制探索
t1 = time.time() # 记录下训练刚开始的时间
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS): # 循环200*200次
        if RENDER:
            env.render()
        
        # 加入探索噪音
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r/10, s_) # pointer在这里+1

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995 # 随机度衰减
            ddpg.learn()

        s = s_ # 更新状态
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, 'Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,)
            # if ep_reward > -300: RENDER = True
            break # 回到上层循环
print('Running time: ', time.time()-t1 )

