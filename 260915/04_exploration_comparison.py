# 큰 탐색 공간에서 탐사 위주와 𝜺-탐욕, UCB의 성능 분석

import numpy as np
import matplotlib.pyplot as plt

def simulate_pull_bandit(handle,visit,win):
    visit[handle]+=1
    win[handle]+=1 if np.random.random()<arm_prob[handle] else 0

def exploration_and_exploitation(rate):
    visit=np.ones(n_arms)  # 당긴 횟수
    win=np.ones(n_arms)    # 승리 횟수
    for t in range(int(n_trial*rate)):  # rate 비율만큼 최적 arm을 알아내는 데 사용
        h=np.random.randint(0,n_arms)
        simulate_pull_bandit(h,visit,win)
    p=np.array([win[i]/visit[i] for i in range(n_arms)])
    h=np.random.choice(np.where(p==np.max(p))[0])
    for t in range(n_trial-int(n_trial*rate)):
        simulate_pull_bandit(h,visit,win)
    return visit,win

def epsilon_greedy(eps):
    visit=np.ones(n_arms)  # 당긴 횟수
    win=np.zeros(n_arms)   # 승리 횟수
    for t in range(n_trial):
        if(np.random.random()<eps):  # 확률 eps로 임의 선택
            h=np.random.randint(0,n_arms)
        else:
            p=np.array([win[i]/visit[i] for i in range(n_arms)])
            h=np.random.choice(np.where(p==np.max(p))[0])
        simulate_pull_bandit(h,visit,win)
    return visit,win

def UCB(c):
    visit=np.ones(n_arms)  # 당긴 횟수
    win=np.ones(n_arms)    # 승리 횟수
    for t in range(n_trial):
        p=np.array([win[i]/visit[i]+c*np.sqrt(np.log(t+1)/visit[i]) for i in range(n_arms)])  # 식 (4.1)
        h=np.random.choice(np.where(p==np.max(p))[0])
        simulate_pull_bandit(h,visit,win)
    return visit,win

def box_plot_analysis(algorithm,hyper_param,alg_name):
    net_profit=[]
    for param in hyper_param:
        net=[]
        for k in range(n_run):  # 박스 플롯 그리기 위해 n_run번 실행
            visit,win=algorithm(param)
            net.append(sum([2*win[i]-visit[i] for i in range(n_arms)]))
        net_profit.append(net)
    print(alg_name+'의 성능:',np.mean(net_profit,axis=1))
    plt.boxplot(net_profit)
    plt.title(alg_name+' performance')
    plt.xlabel('hyper-parameters')
    plt.ylabel('net profits')
    plt.grid()
    plt.xticks(np.arange(1,len(hyper_param)+1),hyper_param)
    plt.show()

n_arms=200  # 손잡이 개수
arm_prob=[0.4+(np.random.random()-0.5)/2 for i in range(n_arms)]  # 밴딧 승률 설정
n_trial=10000  # 손잡이를 당기는 횟수
n_run=50  # 통계 신뢰도를 위해 독립적으로 n_run번 실행

box_plot_analysis(exploration_and_exploitation,[0.0,0.1,0.25,0.5,0.75,0.99,1.0],'exploration_and_exploitation')
box_plot_analysis(epsilon_greedy,[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],'eps-greedy')
box_plot_analysis(UCB,[0.0,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0],'UCB')
