import pandas as pd
import numpy as np
import gym
from gym import spaces
from copy import copy

episode = "EMSRL_IP_2021"


class EMSRLEnv(gym.Env):

    def __init__(self, *args, **kwargs):

        # BRL data import
        self.period = 8760
        self.wind_frac = 0.05
        self.solar_frac = 0.05
        self.ESS_cap = 1500

        self.ESS_eff = 0.95
        self.AWE_eff = 0.68

        self.H2_cost = 6

        self.ESS_P_cap = self.ESS_cap * 0.3
        self.AWE_P_cap = 200

        self.VOM = 30
        self.FOM = 10000
        self.BESS_power_cost = 483000
        self.BESS_capacity_cost = 148000
        self.AWE_cost = 630000
        self.number_of_years = 10
        self.inflation_rate = 0.045
        self.BESS_cos = self.ESS_cap * self.BESS_capacity_cost + self.ESS_P_cap * self.BESS_power_cost
        self.AWE_cos = self.AWE_cost * self.AWE_P_cap
        self.BESS_ann = self.BESS_cos * self.inflation_rate / ((1 + self.inflation_rate) ** self.number_of_years - 1)
        self.AWE_ann = self.AWE_cos * self.inflation_rate / ((1 + self.inflation_rate) ** self.number_of_years - 1)

        data_path = '../dataset/2021_revised.xlsx'
        # if it doesn't work, use your path like data_path = 'C://PycharmProjects/EMSRL/dataset/2020_revised.xlsx'
        df = pd.read_excel(data_path)

        df_wind = df["2021 (Wind) "][0:self.period + 24] * self.wind_frac
        df_solar = df["2021 (Solar) "][0:self.period + 24] * self.solar_frac
        wind_uncertain = np.random.normal(1, 0, size=self.period + 24)
        solar_uncertain = np.random.normal(1, 0, size=self.period + 24)
        df_wind_uncer = df_wind * wind_uncertain
        df_solar_uncer = df_solar * solar_uncertain
        df_PwPs = df_wind_uncer + df_solar_uncer

        self.PwPs = list(np.array(df_PwPs.tolist()))

        self.ELEC_cost = df["elec_price"][24:self.period + 48]
        self.ELEC_cost = self.ELEC_cost.to_numpy()

        #####################################################################################

        self.num_energy = 1  # Number of energy_10 source
        self.ESS_capacity = self.ESS_cap * 0.95  # except ESS min
        self.step_limit = self.period + 24  # Curtailed data period

        self.ESS_cap_remain = copy(self.ESS_capacity)

        self.action_acc = []
        self.ESS_charge = []
        self.ESS_discharge = []
        self.AWE_acc = []
        self.profit = []

        self.penalty = 0

        self.step_sell_reward = 0
        self.total_sell_reward = 0

        self.ELEC_sell = 0
        self.ELEC_store = 0
        self.H2_sell = 0

        self.TBOM = 0
        self.H = 0
        self.TAOM = 0

        # Prices of assets have a mean value in every period and vary according to a Gaussian distribution
        ELECmean = self.ELEC_cost
        asset1mean = ELECmean.reshape(1, -1)
        asset1var = np.ones(asset1mean.shape) * ELECmean * 0
        H2mean = np.ones(self.period + 24) * self.H2_cost
        asset2mean = H2mean.reshape(1, -1)
        asset2var = np.ones(asset2mean.shape) * 4.5 * 0
        self.asset_price_means = np.vstack([asset1mean, asset2mean])
        self.asset_price_var = np.vstack([asset1var, asset2var])

        # Cash on hand, asset prices, num of shares, portfolio value
        self.obs_length = 74

        self.observation_space = spaces.Box(-3000000, 40000000, shape=(self.obs_length,))

        self.action_space = spaces.Box(low=np.array([-self.ESS_P_cap, self.AWE_P_cap * 0.2]),
                                       high=np.array([self.ESS_P_cap, self.AWE_P_cap]), shape=(2,))
        self.reset()

    def _RESET(self):
        self.step_count = 24
        # (self.asset_price_ELEC, self.asset_price_H2) = self._generate_asset_prices()
        self.asset_price_ELEC = self.ELEC_cost
        self.asset_price_H2 = np.ones(self.period + 24) * self.H2_cost
        self.SOC = np.zeros(self.num_energy)
        self.ESS_cap_remain = copy(self.ESS_capacity)
        self.state = np.hstack([
            np.array(self.SOC),
            np.array([self.ESS_cap_remain]),
            self.asset_price_ELEC[self.step_count - 24:self.step_count],
            self.asset_price_H2[self.step_count - 24:self.step_count],
            self.PwPs[self.step_count - 24:self.step_count]
        ])

        self.action_acc = []
        self.ESS_charge = []
        self.ESS_discharge = []
        self.AWE_acc = []
        self.profit = []

        self.step_sell_reward = 0
        self.total_sell_reward = 0

        self.ELEC_sell = 0
        self.ELEC_store = 0
        self.H2_sell = 0

        self.TBOM = 0
        self.H = 0
        self.TAOM = 0

        # self.a = open(f".env_data/action_{episode}.txt", "a")
        # self.b = open(f".env_data/ESS_discharge_{episode}.txt", "a")
        # self.c = open(f".env_data/ESS_charge_{episode}.txt", "a")
        # self.d = open(f".env_data/AWE_sell_{episode}.txt", "a")
        # self.e = open(f".env_data/profit_{episode}.txt", "a")

        return self.state

    def _generate_asset_prices(self):
        asset_prices = np.array([np.random.normal(mu, sig) for mu, sig in
                                 zip(self.asset_price_means.flatten(), self.asset_price_var.flatten())]
                                ).reshape(self.asset_price_means.shape)

        zero_vals = np.vstack(np.where(asset_prices < 0))
        cols = np.unique(zero_vals[0])
        for c in cols:
            first_zero = zero_vals[1][np.where(zero_vals[0] == c)[0].min()]
            asset_prices[c, first_zero:] = 0
        asset_prices_ELEC = asset_prices[0]
        asset_prices_H2 = asset_prices[1]
        return asset_prices_ELEC, asset_prices_H2

    def _STEP(self, action):

        self.action_acc.extend([action])

        assert self.action_space.contains(action)

        ESS_action = action[0]
        AWE_active = action[1]

        if ESS_action == 0:
            binary = 0

        elif ESS_action < 0:
            binary = 0
            ESS_action = -1 * ESS_action

            if ESS_action > self.SOC[0] * self.ESS_eff:
                ESS_action = self.SOC[0] * self.ESS_eff
            else:
                pass

            self.ELEC_sell = ESS_action
            self.ESS_cap_remain += ESS_action / self.ESS_eff
            self.SOC[0] -= ESS_action / self.ESS_eff

            if self.SOC[0] < 0:
                self.SOC[0] = 0

        else:  # z1(discharge) = 0, z2(charge) = 1
            binary = 1

            if ESS_action > self.PwPs[self.step_count]:
                ESS_action = self.PwPs[self.step_count]
            else:
                pass

            if ESS_action > self.ESS_cap_remain / self.ESS_eff:
                ESS_action = self.ESS_cap_remain / self.ESS_eff
            else:
                pass

            self.ELEC_store = ESS_action
            self.ESS_cap_remain -= ESS_action * self.ESS_eff
            self.SOC[0] += ESS_action * self.ESS_eff

        if self.PwPs[self.step_count] - binary * ESS_action < self.AWE_P_cap * 0.2:
            AWE_active = 0
        elif AWE_active > self.PwPs[self.step_count] - binary * ESS_action:
            AWE_active = self.PwPs[self.step_count] - binary * ESS_action
        else:
            pass

        self.H2_sell = AWE_active

        self.step_sell_reward = self.ELEC_sell * self.ELEC_cost[
            self.step_count] + self.H2_sell * self.AWE_eff / 0.0333 * self.H2_cost
        self.total_sell_reward += self.step_sell_reward

        self.TBOM = self.ELEC_store * 30
        self.H = self.H2_sell * self.AWE_eff / 0.0333
        self.TAOM = 10.11 * 0.012 * self.H + 0.0019 * 2.96 * self.H + 0.11 * 0.012 * self.H + 0.00029 * 0.33 * self.H

        reward = (self.step_sell_reward - self.TBOM - self.TAOM - (
                self.BESS_ann + self.AWE_ann + self.ESS_P_cap * self.FOM + 0.05 * self.AWE_P_cap * self.AWE_cost) / self.step_limit) / 100000

        self.step_count += 1

        self.ESS_discharge.extend([self.ELEC_sell])
        self.ESS_charge.extend([self.ELEC_store])
        self.AWE_acc.extend([self.H2_sell])
        self.profit.extend([self.total_sell_reward])

        if self.step_count >= self.step_limit:
            done = True
            # self.a.write("{:s}\n".format(str(self.action_acc)))
            # self.b.write("{:s}\n".format(str(self.ESS_discharge)))
            # self.c.write("{:s}\n".format(str(self.ESS_charge)))
            # self.d.write("{:s}\n".format(str(self.AWE_acc)))
            # self.e.write("{:s}\n".format(str(self.profit)))
            # self.a.close()
            # self.b.close()
            # self.c.close()
            # self.d.close()
            # self.e.close()
        else:
            self._update_state()
            done = False

        return self.state, reward, done, {}

    def _update_state(self):
        self.state = np.hstack([
            np.array(self.SOC),
            np.array([self.ESS_cap_remain]),
            self.asset_price_ELEC[self.step_count - 24:self.step_count],
            self.asset_price_H2[self.step_count - 24:self.step_count],
            self.PwPs[self.step_count - 24:self.step_count]
        ])
        self.step_sell_reward = 0
        self.ELEC_sell = 0
        self.ELEC_store = 0
        self.H2_sell = 0

        self.TBOM = 0
        self.H = 0
        self.TAOM = 0

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()
