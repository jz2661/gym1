from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C,PPO,SAC
import numpy as np
from sim_env import MySim

if __name__ == '__main__':
    env = MySim()
    check_env(env)
    
    #model = A2C (policy="MlpPolicy", env=env)
    model = PPO(policy="MlpPolicy", env=env, verbose=1)
    #model = SAC(policy="MlpPolicy", env=env, verbose=1) # early stop

    model.learn(total_timesteps=50000)
    obs = env.reset()[0]

    #model.save("mysim")
    #model = PPO.load("ppo_cartpole")

    # 验证一次
    for _ in range(100):
        action, state = model.predict(observation=obs) 
        print(f"price: {action[0]}")
        print(f"{env._get_info()}")
        
        obs, reward, done, _, info = env.step(action)
        if done:
            break
