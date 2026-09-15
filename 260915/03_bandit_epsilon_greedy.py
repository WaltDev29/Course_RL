# 다중 손잡이 밴딧에서 순수 탐험과 𝜺-탐욕의 기대 이득 분석

import numpy as np

def simulate_pull_bandit(handle,visit,win):
    visit[handle]+=1
    win[handle]+=1 if np.random.random()<arm_prob[handle] else 0

def exploration(): # 탐험 방식
    visit=np.ones(n_arms) # 당긴 횟수
    win=np.ones(n_arms) # 승리 횟수
    for t in range(n_trial):
        h=np.random.randint(0,n_arms)
        simulate_pull_bandit(h,visit,win)
    return visit,win

def epsilon_greedy(eps): # 𝜺-탐욕
    visit=np.ones(n_arms) # 당긴 횟수
    win=np.ones(n_arms) # 승리 횟수
    for t in range(n_trial):
        if(np.random.random()<eps): # 확률 eps로 임의 선택
            h=np.random.randint(0,n_arms)
        else:
            p=np.array([win[i]/visit[i] for i in range(n_arms)])
            h=np.random.choice(np.where(p==np.max(p))[0])
        simulate_pull_bandit(h,visit,win)
    return visit,win

n_arms=6 # 손잡이 개수(밴딧 기계의 수)
arm_prob=[0.4,0.2,0.1,0.5,0.3,0.6] # 승률
n_trial=10000 # 손잡이를 당기는 횟수

visit,win=exploration()
print('순수 탐험에 의한 통계:')
print(' 승률:', ['%6.4f'% (win[i]/visit[i]) for i in range(n_arms)])
print(' 수익($):',['%d'% (2*win[i]-visit[i]) for i in range(n_arms)])
print(' 순수익($):',sum([2*win[i]-visit[i] for i in range(n_arms)]))

visit,win=epsilon_greedy(0.1)
print('epsilon-탐욕에 의한 통계:')
print(' 승률:', ['%6.4f'% (win[i]/visit[i]) for i in range(n_arms)])
print(' 수익($):',['%d'% (2*win[i]-visit[i]) for i in range(n_arms)])
print(' 순수익($):',sum([2*win[i]-visit[i] for i in range(n_arms)]))
