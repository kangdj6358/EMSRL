<div style="display:flex; align-items: center;">

# Optimal Planning of Hybrid Energy Storage Systems using Curtailed Renewable Energy through Deep Reinforcement Learning

</div>

EMSRL is a reinforcement learning PPO algorithm designed to maximize profits by utilizing battery energy storage systems (BESS) and alkaline water electrolyzers (AWE) to manage curtailed energy generated from solar and wind power.

This model is described in the paper: [Optimal Planning of Hybrid Energy Storage Systems using Curtailed Renewable Energy through Deep Reinforcement Learning](https://arxiv.org/abs/2212.05662)


## Abstract

Energy management systems are becoming increasingly important in order to utilize the continuously growing curtailed
renewable energy. Promising energy storage systems, such as batteries and green hydrogen should be employed to maximize
the efficiency of energy stakeholders. However, optimal decision-making, i.e., planning the leveraging between different
strategies, is confronted with the complexity and uncertainties of large-scale problems. A sophisticated deep
reinforcement learning methodology with a policy-based algorithm is proposed here to achieve real-time optimal energy
storage systems planning under the curtailed renewable energy uncertainty. A quantitative performance comparison proved
that the deep reinforcement learning agent outperforms the scenario-based stochastic optimization algorithm, even with a
wide action and observation space. A robust performance, with maximizing net profit and stable system, was confirmed to
the uncertainty rejection capability of the deep reinforcement learning under a large uncertainty of the curtailed
renewable energy. Action-mapping was performed for visually assessing the action taken by the deep reinforcement
learning agent according to the state. The corresponding results confirmed that the deep reinforcement learning agent
learns the way as the deterministic solution performs, and demonstrates more than 90% profit accuracy compared to the
solution.

-------------------------------

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

-----------------------

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

Datasets related to this article can be found at [CAISO](http://www.caiso.com/informed/Pages/ManagingOversupply.aspx)