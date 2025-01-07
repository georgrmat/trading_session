# Trading Strategies: Bollinger Bands with and without Trend Analysis

This repository contains a Jupyter Notebook showcasing a trading strategy presentation based on Binance 1-hour cryptocurrency data. The goal of this project is to explore and compare trading performance using Bollinger Bands with and without a trend-following filter based on Exponential Moving Averages (EMAs).

---

## Project Overview

In this notebook, we:
1. Load and preprocess Binance 1-hour trading data.
2. Calculate **Exponential Moving Averages (EMAs)** to identify the market trend.
3. Implement and visualize **Bollinger Bands** to define trading ranges and opportunities.
4. Compare two trading approaches:
   - **Bollinger Bands with trend confirmation** (using EMAs to follow the market direction).
   - **Bollinger Bands without trend confirmation** (pure Bollinger Band signals).
5. Evaluate and analyze the results to determine the impact of trend filtering on trading performance.

---

## Content

- **Notebook**: Contains all the code, data analysis, and strategy implementation.
- **Binance 1-hour data**: Historical trading data fetched from Binance for the analysis.
- **Strategy Explanation**:
  - **Bollinger Bands**: A volatility-based indicator consisting of a moving average and two standard deviations away from it.
  - **EMAs**: Used to determine whether the market is trending upwards or downwards.
  - **Trading Rules**:
    - With trend: Only take trades in the direction of the trend as defined by the EMAs.
    - Without trend: Take all Bollinger Band signals regardless of trend direction.

---

## Key Features

- **Data Visualization**: Interactive charts showing the price, EMAs, Bollinger Bands, and trades.
- **Strategy Backtesting**: Performance metrics (e.g., win rate, profit, drawdown) to evaluate the effectiveness of the strategies.
- **Comparative Analysis**: Insights into the pros and cons of incorporating trend-following into Bollinger Band trading.

---

## Getting Started

### Requirements
Make sure you have the following installed:
- Python 3.8+
- Jupyter Notebook
- Required libraries (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - Binance API (if fetching live data)

### Running the Notebook
1. Clone this repository:  
   ```bash
   git clone https://github.com/georgrmat/trading_session.git
   cd trading-strategies-bollinger-bands
   ```
2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
4. Open the `Bollinger_Bands_Trading.ipynb` file to follow along with the analysis.

---

## Results & Observations

At the end of the notebook, you’ll find:
- A comparison of trading performance metrics for both strategies.
- Visualized trades and market conditions for a clearer understanding.
- Insights into when and why incorporating EMAs can improve trading decisions.

---

## Future Work
This project can be extended by:
- Testing on different timeframes (e.g., 15m, 4h).
- Exploring other trend indicators (e.g., MACD, RSI).
- Incorporating additional risk management rules.

---

## Disclaimer
This project is for educational purposes only. Trading cryptocurrencies involves significant risk, and past performance does not guarantee future results. Use this project at your own discretion.

---

## Contributing
Feel free to fork this repository and suggest improvements via pull requests. All contributions are welcome!

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
