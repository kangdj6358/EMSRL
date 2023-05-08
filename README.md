<div style="display:flex; align-items: center;">

# Optimal Planning of Hybrid Energy Storage Systems using Curtailed Renewable Energy through Deep Reinforcement Learning

</div>

EMSRL is a reinforcement learning PPO algorithm designed to maximize profits by utilizing battery energy storage systems (BESS) and alkaline water electrolyzers (AWE) to manage curtailed energy generated from solar and wind power.

This model is described in the paper: [Optimal Planning of Hybrid Energy Storage Systems using Curtailed Renewable Energy through Deep Reinforcement Learning](https://arxiv.org/abs/2212.05662)




## setup

```
setup(
    name="EMSRL",
    version="1.0",
    url="https://github.com/kangdj6358/EMSRL",
    author="Dongju Kang, Doeun Kang",
    license="MIT",
    install_requires=[
        "gym == 0.18.3",
        "ray == 1.9.0",
        "ray[rllib] == 1.9.0",
        "pandas == 1.3.3",
        "openpyxl == 3.0.9",
        "torch == 1.9.1",
    ],
    zip_safe=False,
)
```

## Code implementation example

### Train EMSRL

You can adjust the hyperparameters in the rl_config in the train_EP.py file.

To train the dataset using PPO, please run

```
python EMSRL_train_EP.py
```

### Evaluate the result from checkpoint

After train the data, you can evaluate the results by:

```
python evaluate_EP.py results/{episode}/PPO/PPO_EMSRLEnv_{}/checkpoint_{000000}/checkpoint-{00} --run PPO --env EMSRLEnv --episodes {0000}
```

## Data

Datasets related to this article can be found at [California ISO](http://www.caiso.com/informed/Pages/ManagingOversupply.aspx)