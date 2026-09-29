# Q-러닝의 학습 곡선
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def Q_learning(env,gamma=0.99,rho=0.01,eps=1.0,eps_decay=0.999,eps_min=0.05):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    Qs=[]

    for i in range(10000):
        s,info=env.reset()

        while True:
            if np.random.random()<eps:
                a=np.random.randint(0,env.action_space.n)
            else:
                a=greedy_select(Q[s,:])

            s1,r,terminated,truncated,info=env.step(a)

            if terminated:
                Q[s,a]=Q[s,a]+rho*(r-Q[s,a])
            else:
                Q[s,a]=Q[s,a]+rho*((r+gamma*np.max(Q[s1,:])-Q[s,a]))

            s=s1
            eps=max(eps_min,eps*eps_decay)

            if terminated or truncated:
                break

        Qs.append(Q.copy())

    pi=env.observation_space.n*[None]

    for s in range(env.observation_space.n):
        pi[s]=np.argmax(Q[s])

    return pi,Qs


def learning_curve(Qs):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    colors=['r','g','b','m']

    for s in range(16):
        ax = axes[s]

        ax.grid()
        ax.set_ylim(0, 1.1)
        ax.set_title('state '+str(s), fontsize=16)
        ax.set_ylabel('q value')
        ax.set_xlabel('episodes')

        for a in range(env.action_space.n):
            ax.plot(np.array(Qs)[:,s,a], colors[a])

        ax.legend([0,1,2,3])

    plt.tight_layout()
    plt.show()


env=gym.make(
    'FrozenLake-v1',
    is_slippery=False,
    render_mode='ansi'
)

pi,Qs=Q_learning(env)
env.close()

learning_curve(Qs)