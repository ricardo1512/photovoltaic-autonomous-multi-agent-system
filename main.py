"""
Forecast-Driven Simulation and Visualization of PV-to-Grid Energy Trading Using Negotiation Strategies and Nash Q-Learning.
"""

import argparse
from create_environment import *
from neg_strategies import *
from plot_neg_pairs import *
from nash_q_learning import *
from dynamic_plotting import *
from plot_metrics import *

# Global variables
BATTERY_CAPACITY = 200000
PANEL_AREA = 15000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast-driven simulation for PV-to-grid energy trading.")

    parser.add_argument("--create-env", action="store_true", help="Create environment from weather and market data.")
    parser.add_argument("--start-date", type=str, default="2025-05-01", help="Start date for environment (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2025-05-10", help="End date for environment (YYYY-MM-DD).")
    parser.add_argument("--weather-file", type=str, default="Data/weather_data.csv", help="CSV file with weather forecast.")
    parser.add_argument("--price-file", type=str, default="Data/market_price.csv", help="CSV file with market price.")

    parser.add_argument("--negotiation", action="store_true", help="Run negotiation strategies.")
    parser.add_argument("--nash", action="store_true", help="Run Nash Q-learning.")

    parser.add_argument("--dynamic", action="store_true", help="Generate dynamic plots from results.")
    parser.add_argument("--dynamic-input1", type=str, default="Data/environment.csv",
                        help="Environment input file for dynamic plot.")
    parser.add_argument("--dynamic-input2", type=str, default="Data/results_neg_profit_profit.csv",
                        help="Algorithm results input file for dynamic plot.")

    parser.add_argument("--plot-neg", action="store_true", help="Plot negotiation results.")
    parser.add_argument("--plot-nash", action="store_true", help="Plot Nash history.")
    parser.add_argument("--plot-comparative", action="store_true",
                        help="Generate comparative bar plots for negotiation and Nash.")

    args = parser.parse_args()

    # CREATE ENVIRONMENT
    if args.create_env:
        create_environment(
            panel_area=PANEL_AREA,
            start_date=args.start_date,
            end_date=args.end_date,
            file_weather_forecast=args.weather_file,
            file_market_price=args.price_file,
            output_file="Data/environment.csv"
        )

    # SIMULATIONS
    # 1. Negotiation
    if args.negotiation:
        negotiation_simulation(
            environment_file="Data/environment.csv",
            output_profit_profit="Data/results_neg_profit_profit.csv",
            output_neg_summary="Data/results_neg_summary.csv",
            battery_capacity=BATTERY_CAPACITY,
        )

    # 2. Nash Q-learning
    if args.nash:
        nash_q_learning_simulation(
            environment_file="Data/environment.csv",
            output_nash="Data/results_nash.csv",
            battery_capacity=BATTERY_CAPACITY,
        )

    # PLOTTING
    # 1. Negotiation pairs
    if args.plot_neg:
        negotiation_plot(
            input_file="Data/results_neg_summary.csv",
            output_image="Images/negotiation_results.png"
        )

    # 2. Dynamic Plot (Negotiation or Nash)
    if args.dynamic:
        create_dynamic_plots(
            input_file1=args.dynamic_input1,
            input_file2=args.dynamic_input2
        )

    # 3. Comparative bar plots (Negotiation vs Nash)
    if args.plot_comparative:
        comparative_bar_plot(
            "Images/comparative_neg_nash_results.png",
            "Images/comparative_neg_nash_battery_wear.png"
        )
