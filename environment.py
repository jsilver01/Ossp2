import random
import numpy as np
import matplotlib.pyplot as plt

class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self,a):
        if a == 0:
            self.move_left()
        elif a ==1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a ==3:
            self.move_down()
        
        reward = -1
        done = self.is_done()
        return (self.x,self.y), reward, done
    
    def move_right(self):
        self.y += 1
        if self.y > 3:
            self.y = 3

    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
    
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False
        
    def get_state(self):
        return (self.x , self.y)
    
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent():
    def __init__(self):
        pass

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action

# Dynamic Programming (DP)
def dynamic_planning(data, env, agent, gamma=1.0, theta=0.01):
    while True:
        delta = 0
        for x in range(4):
            for y in range(4):
                v = data[x][y]
                new_v = 0
                for a in range(4):
                    # 새로운 상태, 보상 및 완료 여부 얻기
                    (x_prime, y_prime), reward, _ = env.step(a)
                    # 에이전트가 선택한 행동 사용
                new_v += (1/4) * (reward + gamma * data[x_prime][y_prime])
                data[x][y] = new_v
                delta = max(delta, abs(v - data[x][y]))
        if delta < theta:
            break

# Monte Carlo (MC)
def monte_carlo(data, env, agent, gamma=1.0, alpha=0.01):
    for k in range(50000):
        done = False
        history = []

        while not done:
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x, y, reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            cum_reward = reward + gamma * cum_reward

# n-Step TD Learning
def n_step_td_learning(data, env, agent, Ns, gamma=1.0, alpha=0.01):
    for k in range(50000):
        done = False
        history = []

        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.step(action)
            x_prime, y_prime = env.get_state()
            history.append((x, y, reward, x_prime, y_prime))
        env.reset()

        cum_reward = 0
        T = len(history)
        for t in range(T - Ns):
            G = 0
            for i in range(t, t + Ns):
                x, y, reward, x_prime, y_prime = history[i]
                G += (gamma ** (i - t)) * reward
            if t + Ns < T:
                x, y, _, x_prime, y_prime = history[t + Ns]
                G += (gamma ** Ns) * data[x_prime][y_prime]
            x, y, _, x_prime, y_prime = history[t]
            data[x][y] = data[x][y] + alpha * (G - data[x][y])

def main():
    env = GridWorld()
    agent = Agent()
    Ne_values = [100, 1000, 10000]
    learning_methods = ["DP", "MC", "1-Step TD", "3-Step TD"]

    results = {method: {Ne: [] for Ne in Ne_values} for method in learning_methods}

    for Ne in Ne_values:
        for method in learning_methods:
            data = np.zeros((4, 4))

            if method == "DP":
                dynamic_planning(data, env, agent)
            elif method == "MC":
                monte_carlo(data, env, agent)
            elif "1-Step TD" in method:
                Ns = 1
                n_step_td_learning(data, env, agent, Ns)
            elif "3-Step TD" in method:
                Ns = 3
                n_step_td_learning(data, env, agent, Ns)

            results[method][Ne] = data

    for Ne in Ne_values:
        DP_data = np.array(results["DP"][Ne])
        MC_data = np.array(results["MC"][Ne])
        TD1_data = np.array(results["1-Step TD"][Ne])
        TD3_data = np.array(results["3-Step TD"][Ne])

        # V(s) 값 비교 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(range(16), DP_data.flatten(), label='DP')
        plt.plot(range(16), MC_data.flatten(), label='MC')
        plt.plot(range(16), TD1_data.flatten(), label='1-Step TD')
        plt.plot(range(16), TD3_data.flatten(), label='3-Step TD')
        plt.xlabel('States')
        plt.ylabel('Value')
        plt.title(f'V(s) Comparison for Ne = {Ne}')
        plt.legend()
        plt.show()

        # Bias 및 Variance 계산
        methods_data = [MC_data, TD1_data, TD3_data]
        method_names = ['MC', '1-Step TD', '3-Step TD']

        biases = []
        variances = []
        for data in methods_data:
            biases.append(np.mean(np.abs(data - DP_data)))
            variances.append(np.var(data - DP_data))

        # Bias 및 Variance 막대 그래프
        plt.figure(figsize=(8, 5))
        plt.bar(method_names, biases, color='skyblue')
        plt.xlabel('Methods')
        plt.ylabel('Bias')
        plt.title(f'Bias Comparison for Ne = {Ne}')
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.bar(method_names, variances, color='salmon')
        plt.xlabel('Methods')
        plt.ylabel('Variance')
        plt.title(f'Variance Comparison for Ne = {Ne}')
        plt.show()

if __name__ == '__main__':
    main()