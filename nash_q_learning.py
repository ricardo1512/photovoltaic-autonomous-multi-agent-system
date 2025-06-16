"""
Nash Q-Learning Implementation for Energy Market Trading between Grid and Inverter Agents
This simulation models the interaction between a grid operator and a solar inverter owner using Nash Q-learning
to find optimal energy prices for both sides.
The system includes battery storage, faults and cyberattacks, and various efficiency factors.

Key Components:
- NashQLearningAgent: Implements the Nash Q-learning algorithm for both grid and inverter agents
  It uses a minimax approach when choosing actions, where each agent selects the strategy that maximizes its worst-case outcome.
- Battery: Models battery storage with capacity limits and wear calculation
- Inverter: Handles energy conversion, dealing, and decision-making
"""

import random
import numpy as np
import pandas as pd

# Global variables
BATTERY_INITIAL_PERC = 0.5
BATTERY_MIN = 0.2
BATTERY_MIN_MIN = 0.1
BATTERY_MAX = 0.8
BATTERY_MAX_MAX = 0.9
BATTERY_WEAR = 0.001
EFFICIENCY_TO_GRID = 0.9
EFFICIENCY_TO_BATTERY = 0.97
WINDOW_SIZE = 192  # 2 days
STATES_BATTERY = 5
STATES_ENERGY = 5
STATES_PRICE = 5
EPSILON = 0.2
ALPHA = 0.1
GAMMA = 0.9
Q_INITIAL = 0.1
REWARD = 3

class EnergyAgent:
    def step(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

class NashQLearningAgent(EnergyAgent):
    def __init__(self, actions, epsilon, alpha, gamma, role):
        self.actions = actions  # dict: {'grid':[], 'inv':[]}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.role = role  # 'grid' or 'inv'
        self.Q = {}  # Q-table: state -> {(own_action, opp_action): Q_value}

    def initialize_state(self, state):
        """
        Initialize the Q-table entries for a given state, setting all action pairs' Q-values
        to the initial Q value if the state is not already in the Q-table.
        """
        if state not in self.Q:
            opp_role = 'inv' if self.role == 'grid' else 'grid'
            self.Q[state] = {
                (a_own, a_opp): Q_INITIAL
                for a_own in self.actions[self.role]
                for a_opp in self.actions[opp_role]
            }

    def choose_action(self, state):
        """
        Select an action for this agent's role in the given state using epsilon-greedy
        policy combined with a minimax approach considering worst-case opponent responses.
        """
        self.initialize_state(state)
        opp_role = 'inv' if self.role == 'grid' else 'grid'
        opp_actions = self.actions[opp_role]

        if random.random() < self.epsilon:
            return random.choice(self.actions[self.role])

        best_action = None
        best_value = -np.inf
        for a_own in self.actions[self.role]:
            # Assume opponent actions uniformly random
            q_values = [self.Q[state][(a_own, a_opp)] for a_opp in opp_actions]
            # worst-case value for own action
            worst_value = min(q_values)
            if worst_value > best_value:
                best_value = worst_value
                best_action = a_own
        return best_action

    def update_Q(self, state, a_own, a_opp, reward, next_state):
        """
        Update the Q-table value for the current state-action pair using the Nash Q-learning
        update rule that considers the best response to the opponent's worst-case action.
        """
        self.initialize_state(state)
        self.initialize_state(next_state)
        q_current = self.Q[state][(a_own, a_opp)]

        opp_role = 'inv' if self.role == 'grid' else 'grid'
        opp_actions = self.actions[opp_role]

        # Calculate value for next state as max over own actions of min opponent Q-value
        next_vals = []
        for a_own_p in self.actions[self.role]:
            min_val = min(self.Q[next_state][(a_own_p, a_opp_p)] for a_opp_p in opp_actions)
            next_vals.append(min_val)
        v_next = max(next_vals)

        self.Q[state][(a_own, a_opp)] = q_current + self.alpha * (reward + self.gamma * v_next - q_current)

    @staticmethod
    def find_actions(state, agent_inv, agent_grid):
        """
        Static helper method to find actions chosen by inverter and grid agents independently
        in the given state, assuming opponents choose uniformly random actions.
        """
        a_inv = agent_inv.choose_action(state)
        a_grid = agent_grid.choose_action(state)
        return a_inv, a_grid


class Battery(EnergyAgent):
    def __init__(self, capacity):
        self.capacity = capacity
        self.stored = capacity * BATTERY_INITIAL_PERC
        self.total_wear = 0.0
        self.min_perc = BATTERY_MIN
        self.max_perc = BATTERY_MAX
        self.min_limit = capacity * self.min_perc
        self.max_limit = capacity * self.max_perc

    def store_energy(self, amount):
        """
        Attempt to store energy in the battery up to its capacity limits.
        """
        space = self.max_limit - self.stored
        e_store = min(amount, space)
        self.stored += e_store
        if self.stored < self.capacity * self.min_perc:
            return 0.0, 0.0
        e_lost = amount - e_store
        return e_store, e_lost

    def withdraw_energy(self, amount, price):
        """
        Withdraw energy from the battery while accounting for wear caused by energy use.
        """
        max_withdrawable = max(0.0, self.stored - self.min_limit)
        energy_withdrawn = min(max_withdrawable, amount)
        self.stored -= energy_withdrawn
        wear = energy_withdrawn * price * BATTERY_WEAR
        self.total_wear += wear
        if self.stored < self.capacity * self.min_perc:
            return 0.0, 0.0
        return energy_withdrawn, wear

    def get_total_wear(self):
        """
        Get the total accumulated wear on the battery.
        """
        return self.total_wear

class Inverter(EnergyAgent):
    def __init__(self, inv_agent, eff_grid=EFFICIENCY_TO_GRID, eff_batt=EFFICIENCY_TO_BATTERY):
        self.inv_agent = inv_agent
        self.eff_grid = eff_grid
        self.eff_batt = eff_batt
        self.battery = None
        self.grid_agent = None
        self.forecast = 0.0
        self.price_history = []

    def connect(self, battery, grid_agent):
        """
        Connect the inverter to a battery instance and the grid agent for trading.
        """
        self.battery = battery
        self.grid_agent = grid_agent

    def set_forecast(self, f_cast):
        """
        Set the current energy forecast value for inverter decisions.
        """
        self.forecast = f_cast

    def update_price_history(self, price):
        """
        Append a new price to the price history, maintaining a fixed window size.
        """
        self.price_history.append(price)
        if len(self.price_history) > WINDOW_SIZE:
            self.price_history.pop(0)

    def get_historical_avg_price(self):
        """
        Calculate the average price from the price history.
        """
        return np.mean(self.price_history) if self.price_history else 0.0

    def sell_to_grid(self, energy):
        """
        Calculate effective energy sold to the grid after accounting for efficiency losses.
        """
        return energy * self.eff_grid

    def store_energy_battery(self, energy):
        """
        Store energy in the battery after applying battery efficiency loss.
        """
        effective_energy = energy * self.eff_batt
        return self.battery.store_energy(effective_energy)

    def process_sale(self, target_energy, price):
        """
        Process sale of energy by splitting between grid and battery, and calculating profits.
        """
        e_grid = self.sell_to_grid(min(target_energy, self.forecast))
        energy_needed_from_battery = max(0.0, target_energy - self.forecast)
        battery_sell, wear = self.battery.withdraw_energy(energy_needed_from_battery, price)
        sold = e_grid + battery_sell
        return sold, e_grid, battery_sell, wear

    def settle_agreement(self, price_forecast, state, cyberattack):
        """
        Settle a pricing agreement between inverter and grid agents based on the current state
        and a forecasted energy price. Adjusts battery operating limits and updates Q-tables
        based on agreement success or failure.
        """
        self.update_price_history(price_forecast)
        avg_price = self.get_historical_avg_price()

        a_i, a_g = NashQLearningAgent.find_actions(state, self.inv_agent, self.grid_agent)

        print(f"[DEALING] State: {state} | Forecast Price: {price_forecast:.2f} | Grid Action: {a_g:.2f} | Inv Action: {a_i:.2f} | Avg Price: {avg_price:.2f}")

        if cyberattack:
            return False, a_i, a_g

        if price_forecast > avg_price:
            self.battery.min_perc = BATTERY_MIN_MIN
            self.battery.max_perc = BATTERY_MAX_MAX
        else:
            self.battery.min_perc = BATTERY_MIN
            self.battery.max_perc = BATTERY_MAX

        self.battery.min_limit = self.battery.capacity * self.battery.min_perc
        self.battery.max_limit = self.battery.capacity * self.battery.max_perc

        if a_i <= price_forecast and a_g <= price_forecast:
            r_inv = REWARD
            r_grid = REWARD
            self.grid_agent.update_Q(state, a_g, a_i, r_grid, state)
            self.inv_agent.update_Q(state, a_i, a_g, r_inv, state)
            return True, a_i, a_g
        else:
            r_inv = -REWARD if a_i > price_forecast else REWARD
            r_grid = -REWARD if a_g > price_forecast else REWARD
            self.grid_agent.update_Q(state, a_g, a_i, r_grid, state)
            self.inv_agent.update_Q(state, a_i, a_g, r_inv, state)
            return None, a_i, a_g

    def run_step(self, max_energy_forecast, max_price, energy_forecast, price_forecast, pv_fault=0, cyberattack=0, buy_percent=1.0):
        """
        Execute a simulation step including energy forecast handling, price negotiation,
        cyberattack management, energy trading, battery operations, and profit calculation.
        """
        self.set_forecast(energy_forecast)
        battery_state = int((self.battery.stored / self.battery.capacity) * STATES_BATTERY)
        price_state = int(max(0.0, price_forecast / max_price) * STATES_PRICE)
        energy_forecast_state = int(max(0.0, energy_forecast / max_energy_forecast) * STATES_ENERGY)

        state = (battery_state, price_state, energy_forecast_state)

        if pv_fault:
            lost = energy_forecast * price_forecast
            print(f"[FAULT] PV Fault! Forecast lost: {lost:.2f}")
            return 0.0, 0.0, lost, 0.0, self.battery.get_total_wear(), self.battery.stored

        trade, a_i, a_g = self.settle_agreement(price_forecast, state, cyberattack)

        profit_inv = profit_grid = lost = pct_buy = 0.0

        if energy_forecast <= 0:
            e_store, e_lost = self.battery.store_energy(0)
        else:
            battery_available = max(0.0, self.battery.stored - self.battery.min_limit)
            total_available = energy_forecast + battery_available
            energy_to_sell = total_available * buy_percent

            if cyberattack:
                print(f"[CYBER] Cyberattack detected. Storing entire energy to battery.")
                e_store, e_lost = self.store_energy_battery(energy_forecast)
                lost = e_lost * price_forecast

            elif trade:
                sold, e_grid, battery_sell, wear = self.process_sale(energy_to_sell, price_forecast)
                sold_price = sold * price_forecast

                if np.isclose(a_i, 0.0) and np.isclose(a_g, 0.0):
                    inv_part = 0.5
                    grid_part = 0.5
                else:
                    inv_part = a_i / (a_i + a_g)
                    grid_part = a_g / (a_i + a_g)

                profit_inv = sold_price * inv_part
                profit_grid = sold_price * grid_part

                print(f"[TRADE] Sold Energy | Inv Gain: {profit_inv:.2f} | Grid Gain: {profit_grid:.2f}")

                pct_buy = buy_percent

            else:
                print("[TRADE] Dealing failed. Storing energy in the battery.")
                e_store, e_lost = self.store_energy_battery(energy_forecast)
                lost = e_lost * price_forecast

        return profit_inv, profit_grid, lost, pct_buy, self.battery.get_total_wear(), self.battery.stored


def nash_q_learning_simulation(environment_file, output_nash, battery_capacity):
    """
    Run the Nash Q-learning simulation over time using data from an environment CSV file,
    iteratively processing forecasts, faults, trades, and updating agents.
    """
    df = pd.read_csv(environment_file)
    max_energy_forecast = max(df["E_inverter_input_forecast"])
    max_price_forecast = max(df["Price_forecast"])

    # Define actions as integer prices from 0 to max_price_forecast
    grid_price_range = list(range(int(max_price_forecast) + 1))
    inv_price_range = list(range(int(max_price_forecast) + 1))
    actions = {'grid': grid_price_range, 'inv': inv_price_range}

    grid_agent = NashQLearningAgent(actions, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, role='grid')
    inv_agent = NashQLearningAgent(actions, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, role='inv')

    battery = Battery(battery_capacity)
    inverter = Inverter(inv_agent)
    inverter.connect(battery, grid_agent)

    dates = []
    grid_profits = []
    inverter_profits = []
    grid_profit_step_list = []
    inverter_profit_step_list = []
    lost_energy_cumulative = []
    lost_energy_value_step_list = []
    wear_cumulative = []
    buy_pct_list = []
    battery_pct_list = []

    cum_grid = 0.0
    cum_inv = 0.0
    cum_lost = 0.0

    for i, row in df.iterrows():
        energy_forecast = row['E_inverter_input_forecast']
        price_forecast = row['Price_forecast']
        pv_fault = row['PV_fault']
        cyberattack = row['Cyberattack_alert']
        buy_percent = row['Buy_percentage']
        date = pd.to_datetime(row['date'])
        print(f"\n[STEP {i}] Date: {date} | Energy Forecast: {energy_forecast:.2f} | Price Forecast: {price_forecast:.2f} | Buy %: {buy_percent:.2f}")

        gain_inv, gain_grid, lost, pct_buy, wear, stored = inverter.run_step(
            max_energy_forecast, max_price_forecast, energy_forecast, price_forecast, pv_fault=pv_fault, cyberattack= cyberattack, buy_percent=buy_percent
        )
        print(f"[RESULTS] Grid Profit Step: {gain_grid:.2f} | Inv Profit Step: {gain_inv:.2f} | Lost Energy: {lost:.2f} | Battery %: {(stored / battery.capacity) * 100:.2f} | Wear: {wear:.4f}")

        cum_inv += gain_inv
        cum_grid += gain_grid
        cum_lost += lost

        dates.append(date)
        grid_profits.append(cum_grid)
        inverter_profits.append(cum_inv)
        grid_profit_step_list.append(gain_grid)
        inverter_profit_step_list.append(gain_inv)
        lost_energy_cumulative.append(cum_lost)
        lost_energy_value_step_list.append(lost)
        wear_cumulative.append(wear)
        buy_pct_list.append(pct_buy)
        battery_pct_list.append((stored / battery.capacity) * 100)

    results_df = pd.DataFrame({
        'date': dates,
        'Grid_profits_cumulative': grid_profits,
        'Inverter_profits_cumulative': inverter_profits,
        'Grid_profits_step': grid_profit_step_list,
        'Inverter_profits_step': inverter_profit_step_list,
        'Lost_energy_cumulative': lost_energy_cumulative,
        'Lost_energy_value_step': lost_energy_value_step_list,
        'Battery_wear_cumulative': wear_cumulative,
        'Battery_pct': battery_pct_list,
    })

    results_df.to_csv(output_nash, index=False)
    print(f"\nNASH Q-LEARNING: Simulation complete. Results saved to {output_nash}")