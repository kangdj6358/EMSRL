import ray
from EMSRL_env_EP import BRLEnv as env
from ray import tune

episode = "RT_EP_2"

def register_env(env_name, env_config={}):
    # env = create_env(env_name)
    tune.register_env(env_name,
                      lambda env_name: env(env_name,
                                           env_config=env_config))

env_name = 'BRLEnv'
env_config = {}  # Change environment parameters here
rl_config = dict(
    env=env_name,
    num_workers=1,
    env_config=env_config,
    lr=3e-5,
    framework='torch',
    train_batch_size=8000,
    model = dict(
        fcnet_hiddens=[256,256,256],
    )
)

# Register environment
register_env(env_name, env_config)

# Initialize Ray and Build Agent
ray.init(ignore_reinit_error=True)

# analysis = tune.run("PPO", config=rl_config, stop=stopper, verbose=1, local_dir=f'~/ray_results/BRL/ppo_{episode}', checkpoint_freq=10)
analysis = tune.run("PPO", config=rl_config, verbose=1, local_dir=f'~/results/checkpoints/{episode}', checkpoint_freq=5)
# print("Best hyperparameters found were: ", analysis.best_config)

ray.shutdown()
