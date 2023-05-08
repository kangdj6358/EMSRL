import os

episode = "2021_10_IP"

for i in range(1000-856):

    fr = open(f"./env_{episode}.py", 'r')
    lines = fr.readlines()
    fr.close()

    fw = open(f"./env_{episode}.py", "w")
    for line in lines:
        if f'Data_path = "./energy2021_10/energy{i+856}.csv"' in line:
            fw.write(f'Data_path = "./energy2021_10/energy{i+1+856}.csv"\n')
        else:
            fw.write(line)
    fw.close()

    commend = f'python evaluate_{episode}.py /home/kangdj6358/ray_results/paper/rl_1/PPO/PPO_BRLEnv_9cb80_00000_0_2022-01-16_20-36-57/checkpoint_001425/checkpoint-1425 --run PPO --episodes 1 --env BRLEnv'
    os.system(commend)