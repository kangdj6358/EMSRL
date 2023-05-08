import os

episode = "2021_10_EP"

for i in range(1000-999):

    fr = open(f"./env_{episode}.py", 'r')
    lines = fr.readlines()
    fr.close()

    fw = open(f"./env_{episode}.py", "w")
    for line in lines:
        if f'Data_path = "./energy2021_10/energy{i+999}.csv"' in line:
            fw.write(f'Data_path = "./energy2021_10/energy{i+1+999}.csv"\n')
        else:
            fw.write(line)
    fw.close()

    commend = f'python evaluate_{episode}.py /home/kangdj6358/ray_results/paper/rl_1001/PPO/PPO_BRLEnv_e49f3_00000_0_2022-01-16_18-01-29/checkpoint_001490/checkpoint-1490 --run PPO --episodes 1 --env BRLEnv'
    os.system(commend)