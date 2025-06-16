# PV-to-Grid Energy Trading Simulator Using a Multi-Agent System
### Forecast-driven simulation using negotiation strategies and Nash Q-learning.

#### 1. Overview 
This project simulates photovoltaic (PV) energy trading with the grid, 
using weather forecasts, market prices, faults, and cyberattacks. It supports negotiation strategies 
and reinforcement learning (Nash Q-learning, minimax) to optimize profit and 
energy efficiency. It was developed by Ricardo Vicente (Group 14) as part of 
a project for the Autonomous Agents and Multi-Agent Systems course in 2025 at 
Instituto Superior Técnico, University of Lisbon.

#### 2. Features
- Create environment from weather, faults, and market data
- Run negotiation-based trading strategies
- Train agents using Nash Q-learning and minimax
- Visualize results dynamically or statically

#### 3. Requirements 
Install dependencies:
```bash
pip install numpy pandas pyqtgraph PyQt5 requests-cache retry-requests
``` 
#### 4. Run
###### Environment
⚠️ Note: Since [REN - Market Prices](https://mercado.ren.pt/PT/Electr/InfoMercado/InfOp/MercOmel/Paginas/Precos.aspx) does not provide API access to market data, the original .xlsx file must be 
manually downloaded, edited, and converted to .csv before use.
```bash
python main.py --create-env
``` 
Output: ```Data/environment.csv```

###### Negotiation Simulation (All strategy pairs and specifically for Profit-Profit)
```bash
python main.py --negotiation
```
Output: ```Data/results_neg_profit_profit.csv``` and ```Data/results_neg_summary.csv```

######  Plot Negotiation Results
```bash
python main.py --plot-neg
```
Output: ```Images/negotiation_results.png```

######  Nash Q-learning Simulation
```bash
python main.py --nash
```
Output: ```Data/results_nash.csv```

######  Generate Dynamic Plot (Environment and Results)
- For Negotiation
```bash
python main.py --dynamic --dynamic-input1 Data/environment.csv   --dynamic-input2 Data/results_neg_profit_profit.csv
```
Output: A dynamic plot will be displayed.

- For Nash Q-learning
```bash
python main.py --dynamic  --dynamic-input1 Data/environment.csv   --dynamic-input2 Data/results_nash.csv
```
Output: A dynamic plot will be displayed.

######  Generate Two Comparative Bar Plots (Negotiation vs Nash)
```bash
python main.py --plot-comparative
```
Output: ```Images/comparative_neg_nash_results.png``` and ```Images/comparative_neg_nash_battery_wear.png```

#### 5. Output Files

- `Data/environment.csv`: Results of negotiation.
- `Data/results_nash.csv`: Nash Q-learning results.
- `Data/results_neg_profit_profit.csv`: Profit-Profit negotiation results.
- `Data/results_neg_summary.csv`: Results of all negotiation strategies.
- `Images/comparative_neg_nash_battery_wear.png`: Comparative results visualization of battery wear.
- `Images/negotiation_results.png`: Visualization of Comparative Results for Negotiation Strategies.
