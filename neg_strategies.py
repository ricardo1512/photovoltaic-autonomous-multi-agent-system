"""
Energy Negotiation Simulation between Photovoltaic System and External Grid using Multiple Behavioural Modes

This simulation models energy transactions between a photovoltaic (PV) system—comprising an inverter and battery—
and the external power grid. It accounts for varying negotiation strategies, efficiency losses, battery wear,
operational faults, and potential cyberattacks.

Key Features:
- Grid and Inverter Agents: Each can operate under different negotiation strategies (fair, profit-driven, yes-man)
- Battery Model: Tracks capacity, wear over time, and energy storage/release limits
- Cyberattacks: Responds to abnormal pricing behaviour
- Decision Logic: Dynamic pricing acceptance and battery behaviour based on historical trends
- Performance Logging: Tracks and saves data on financial profits, energy losses, battery wear, and grid-inverter interactions
"""

import random
import pandas as pd
import numpy as np

# Global variables
BATTERY_INITIAL_PERC = 0.5
BATTERY_MIN = 0.2
BATTERY_MIN_MIN = 0.1
BATTERY_MAX = 0.8
BATTERY_MAX_MAX = 0.9
BATTERY_WEAR = 0.001
BATTERY_DECIDE = 1.3
YES_MAN, FAIR, PROFIT = 10, 5, 2
FAIR_PERCENT = 0.4
PROFIT_PERCENT = 0.6
EFFICIENCY_TO_GRID = 0.9
EFFICIENCY_TO_BATTERY = 0.97
WINDOW_SIZE = 192 # 2 days

class EnergyAgent:
    def step(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError


class Grid:
    def __init__(self, negotiation_mode='fair'):
        self.total_energy_received = 0.0
        self.total_earned_money = 0.0
        self.negotiation_mode = negotiation_mode

    def negotiate(self, inverter_offer, price_forecast, cyberattack):
        """
        Attempt to agree on a price with the inverter, iteratively increasing the offer
        depending on the negotiation strategy. Cyberattack conditions may apply.
        """
        if self.negotiation_mode == 'yes-man':
            price = max(price_forecast - YES_MAN, 1)
            max_price = price_forecast
            step = 1
        elif self.negotiation_mode == 'fair':
            price = max(price_forecast - FAIR, 1)
            max_price = price_forecast
            step = 0.75
        else: #                        'profit'
            price = max(price_forecast - PROFIT, 1)
            max_price = price_forecast
            step = 0.5

        accepted_price = None

        while price <= max_price:
            if cyberattack:
                offer_price = price_forecast * random.uniform(1.01, 5)
            else:
                offer_price = price

            accepted = inverter_offer(offer_price)

            print(f"Grid proposed € {offer_price:.2f} -> Inverter {'accepted' if accepted else 'rejected'}")

            if accepted:
                accepted_price = offer_price
                break

            price += step

        return accepted_price

    def receive(self, amount, price_buy, price_forecast):
        """
        Register energy received from the inverter, calculate profit.
        """
        self.total_energy_received += amount
        earned = amount * (price_forecast - price_buy)
        self.total_earned_money += earned
        print(f"Grid bought {amount:.2f} Wh for € {price_buy:.2f}, profit: € {earned:.2f}")

    def get_state(self):
        """
        Return total energy received by the grid.
        """
        return self.total_energy_received

    def get_total_earned(self):
        """
        Return total profits earned by the grid.
        """
        return self.total_earned_money


class Battery(EnergyAgent):
    def __init__(self, capacity):
        self.capacity = capacity
        self.stored = capacity * BATTERY_INITIAL_PERC  # initial battery level 50%
        self.total_wear = 0.0

    def store_energy(self, amount):
        """
        Store incoming energy in the battery.
        """
        space = self.capacity - self.stored
        energy_to_store = min(amount, space)
        self.stored += energy_to_store
        energy_lost = amount - energy_to_store
        return energy_to_store, energy_lost

    def withdraw_energy(self, amount, price, battery_min):
        """
        Withdraw energy from the battery, respecting minimum energy threshold and calculating wear.
        """
        min_battery_limit = self.capacity * battery_min
        max_withdrawable = max(0.0, self.stored - min_battery_limit)
        energy_withdrawn = min(max_withdrawable, amount)
        self.stored -= energy_withdrawn
        wear = energy_withdrawn * price * BATTERY_WEAR
        self.total_wear += wear
        return energy_withdrawn, wear

    def get_state(self):
        """
        Return the current energy stored in the battery.
        """
        return self.stored


class Inverter(EnergyAgent):
    def __init__(self, efficiency_to_grid=EFFICIENCY_TO_GRID, efficiency_to_battery=EFFICIENCY_TO_BATTERY, negotiation_mode='fair'):
        self.efficiency_to_grid = efficiency_to_grid
        self.efficiency_to_battery = efficiency_to_battery
        self.energy_forecast = 0.0
        self.battery = None
        self.grid = None
        self.total_earned_money = 0.0
        self.negotiation_mode = negotiation_mode
        self.total_lost_energy = 0.0
        self.last_buy_pct = None
        self.buy_pct_history = []
        self.price_history = []

    def connect(self, battery, grid):
        """
        Connect the inverter to its battery and the external grid.
        """
        self.battery = battery
        self.grid = grid

    def set_forecast(self, forecast):
        """
        Set the forecasted solar energy production for the current time step.
        """
        self.energy_forecast = forecast

    def negotiate_with_grid(self, price_forecast, cyberattack, buy_pct):
        """
        Manage the energy sale process: evaluate the market, decide on storage vs sale,
        manage cyberattacks, and execute deals or fallback actions.
        """
        self.price_history.append(price_forecast)
        recent_prices = self.price_history[-WINDOW_SIZE:]
        historical_avg_price = np.mean(recent_prices)

        panel_energy = self.energy_forecast
        battery_energy = self.battery.stored
        total_energy = panel_energy + battery_energy

        # Battery exceptions:
        # Can go down to 10% if purchase price is "very high" and worth selling
        # Can go up to 90% if price is very low to sell, waiting for better value
        can_go_down_to_10 = False
        can_go_up_to_90 = False

        # Define "very high" and "very low" prices
        if price_forecast > BATTERY_DECIDE * historical_avg_price:
            can_go_down_to_10 = True
            can_go_up_to_90 = True

        # Adjust battery limits
        battery_min = BATTERY_MIN_MIN if can_go_down_to_10 else BATTERY_MIN
        battery_max = BATTERY_MAX_MAX if can_go_up_to_90 else BATTERY_MAX

        battery_energy_min = battery_min * self.battery.capacity
        battery_energy_max = battery_max * self.battery.capacity

        panel_energy_for_sale = panel_energy
        battery_energy_for_sale = max(0.0, battery_energy - battery_energy_min)
        total_energy_offered = panel_energy_for_sale + battery_energy_for_sale

        if not (total_energy_offered > 0):
            print("No energy available for sale at this time. Profits zeroed.")
            return None

        self.last_buy_pct = buy_pct
        self.buy_pct_history.append(buy_pct)
        purchased_energy = total_energy * buy_pct

        def inverter_offer(price_offer):
            """
            Determine if offer price is acceptable.
            """
            if cyberattack:
                print("!!! CYBERATTACK ALERT: Abusive offer detected and refused !!!")
                return False

            if self.negotiation_mode == "yes-man":
                return True
            elif self.negotiation_mode == "fair":
                return price_offer >= FAIR_PERCENT * price_forecast
            elif self.negotiation_mode == "profit":
                return price_offer >= PROFIT_PERCENT * price_forecast

            return False

        negotiated_price = self.grid.negotiate(inverter_offer, price_forecast, cyberattack)

        if cyberattack:
            # No negotiation or profits
            remaining_energy = panel_energy

            battery_energy_max = battery_max * self.battery.capacity

            battery_space = battery_energy_max - self.battery.stored
            stored_energy = min(remaining_energy * self.efficiency_to_battery, max(0, battery_space))
            self.battery.stored += stored_energy

            lost_energy = max(0, remaining_energy - stored_energy)
            lost_value = lost_energy * price_forecast
            self.total_lost_energy += lost_value

            print(f"Cyberattack: No sale made. Stored energy: {stored_energy:.2f} Wh")
            print(f"Energy lost this step: {lost_energy:.4f} Wh, cost: € {lost_value:.2f}")

            # Exit function without selling and without updating profits
            return

        if negotiated_price is not None:
            panel_energy_sold = min(purchased_energy, panel_energy)
            battery_energy_sold = max(0.0, purchased_energy - panel_energy_sold)

            panel_energy_sold_effective = panel_energy_sold * self.efficiency_to_grid
            battery_energy_sold_effective = battery_energy_sold * self.efficiency_to_grid

            battery_energy_withdrawn, wear = self.battery.withdraw_energy(battery_energy_sold, negotiated_price, battery_min)

            total_energy_sold_effective = panel_energy_sold_effective + battery_energy_sold_effective
            self.grid.receive(total_energy_sold_effective, negotiated_price, price_forecast)

            inverter_profit = total_energy_sold_effective * negotiated_price
            self.total_earned_money += inverter_profit

            print(
                f"Inverter sold {total_energy_sold_effective:.2f} Wh for € {negotiated_price:.2f}, profit: € {inverter_profit:.2f}")
            print(f"Battery wear this sale: € {wear:.4f}")

            remaining_energy = max(0.0, panel_energy - panel_energy_sold)

            battery_space = battery_energy_max - self.battery.stored
            stored_energy = min(remaining_energy * self.efficiency_to_battery, max(0, battery_space))
            self.battery.stored += stored_energy

            lost_energy = remaining_energy - stored_energy
            if lost_energy < 0:
                lost_energy = 0
            self.total_lost_energy += lost_energy * price_forecast

            print(f"Energy stored in battery: {stored_energy:.2f} Wh")
            print(f"Energy lost this step: {lost_energy:.4f} Wh")


        else:
            remaining_energy = panel_energy

            battery_energy_max = battery_max  * self.battery.capacity
            battery_space = battery_energy_max - self.battery.stored

            energy_that_can_be_stored = min(remaining_energy, battery_space / self.efficiency_to_battery)

            stored_energy = energy_that_can_be_stored * self.efficiency_to_battery
            self.battery.stored += stored_energy

            lost_energy = remaining_energy - energy_that_can_be_stored
            self.total_lost_energy += lost_energy * price_forecast

            print(f"Not worth selling. Energy stored in battery: {stored_energy:.2f} Wh")
            print(f"Energy lost this step: {lost_energy:.4f} €")

    def step(self, price_forecast, cyberattack, buy_pct):
        """
        Run a complete decision-making and negotiation cycle for the current timestep.
        """
        print(f"Inverter received forecast of {self.energy_forecast:.2f} Wh with predicted price € {price_forecast:.2f}")
        self.negotiate_with_grid(price_forecast, cyberattack, buy_pct)

    def get_state(self):
        """
        Return the current energy forecast (i.e., predicted solar energy input).
        """
        return self.energy_forecast

    def get_money_earned(self):
        """
        Return total money earned by the inverter over time.
        """
        return self.total_earned_money

    def get_energy_lost(self):
        """
        Return total monetary loss due to unused or excess energy.
        """
        return self.total_lost_energy

    def get_total_wear(self):
        """
        Return cumulative wear cost of the battery.
        """
        return self.battery.total_wear if self.battery else 0.0

    def get_buy_pct_history(self):
        """
        Return the list of buy percentage values used in each timestep.
        """
        return self.buy_pct_history

def negotiation_simulation(environment_file, output_profit_profit, output_neg_summary, battery_capacity):
    """
    Run a full simulation over time using energy and price forecast data.
    Tries all combinations of inverter/grid strategies, logs outputs,
    and saves detailed results and summary.
    """
    negotiation_modes = ["fair", "yes-man", "profit"]

    df = pd.read_csv(environment_file)
    energy_forecasts = df['E_inverter_input_forecast'].tolist()
    price_forecasts = df['Price_forecast'].tolist()
    buy_percentages = df['Buy_percentage'].tolist()
    cyberattack_flags = df['Cyberattack_alert'].tolist()

    results_summary = []

    for grid_mode in negotiation_modes:
        for inverter_mode in negotiation_modes:
            print(f"\n===== Simulation: Grid={grid_mode}, Inverter={inverter_mode} =====")

            inverter = Inverter(negotiation_mode=inverter_mode)
            battery = Battery(capacity=battery_capacity)
            grid = Grid(negotiation_mode=grid_mode)
            inverter.connect(battery, grid)

            grid_profits = []
            inverter_profits = []
            grid_profit_step_list = []
            inverter_profit_step_list = []
            lost_energy_cumulative = []
            lost_energy_value_step_list = []
            wear_cumulative = []
            buy_pct_list = []
            battery_pct_list = []
            dates = []

            for i, (energy_forecast, price, cyberattack, buy_pct) in enumerate(zip(energy_forecasts, price_forecasts, cyberattack_flags, buy_percentages)):
                dates.append(df.loc[i, 'date'])
                print(f"\n[STEP {i}] Date: {df.loc[i, 'date']} | Energy Forecast: {energy_forecast:.2f} | Price Forecast: {price:.2f} | Buy %: {buy_pct:.2f}")
                if df.loc[i, 'PV_fault'] == 1:
                    inverter.total_lost_energy += energy_forecast * price
                    inverter.set_forecast(0.0)
                else:
                    inverter.set_forecast(energy_forecast)
                    inverter.step(price_forecast=price, cyberattack=cyberattack, buy_pct=buy_pct)

                grid_profit_total = grid.get_total_earned()
                inverter_profit_total = inverter.get_money_earned()

                if i == 0:
                    grid_profit_step = grid_profit_total
                    inverter_profit_step = inverter_profit_total
                    lost_money = inverter.get_energy_lost()
                else:
                    grid_profit_step = grid_profit_total - grid_profits[-1]
                    inverter_profit_step = inverter_profit_total - inverter_profits[-1]
                    lost_money = inverter.get_energy_lost() - lost_energy_cumulative[-1]

                battery_pct = (battery.stored / battery.capacity) * 100

                grid_profits.append(grid_profit_total)
                inverter_profits.append(inverter_profit_total)
                grid_profit_step_list.append(grid_profit_step)
                inverter_profit_step_list.append(inverter_profit_step)
                lost_energy_cumulative.append(inverter.get_energy_lost())
                lost_energy_value_step_list.append(lost_money)
                wear_cumulative.append(inverter.get_total_wear())
                buy_pct_list.append(buy_pct)
                battery_pct_list.append(battery_pct)

            if grid_mode == "profit" and inverter_mode == "profit":
                results_df = pd.DataFrame({
                    'date': dates,
                    'Grid_profits_cumulative': grid_profits,
                    'Inverter_profits_cumulative': inverter_profits,
                    'Grid_profits_step': grid_profit_step_list,
                    'Inverter_profits_step': inverter_profit_step_list,
                    'Lost_energy_cumulative': lost_energy_cumulative,
                    'Lost_energy_value_step': lost_energy_value_step_list,
                    'Battery_wear_cumulative': wear_cumulative,
                    'Buy_pct': buy_pct_list,
                    'Battery_pct': battery_pct_list,
                })
                results_df.to_csv(output_profit_profit, index=False)

            summary = {
                'grid_mode': grid_mode,
                'inverter_mode': inverter_mode,
                'grid_profits': grid.get_total_earned(),
                'inverter_profits': inverter.get_money_earned(),
                'lost_energy_value': inverter.get_energy_lost(),
                'battery_wear': inverter.get_total_wear(),
            }
            results_summary.append(summary)

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_neg_summary, index=False)
    print(f"\nNEGOTIATION STRATEGIES: Simulation complete. Results saved to {output_profit_profit} and {output_neg_summary}")
