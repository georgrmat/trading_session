import pandas as pd
import numpy as np


def calculate_ema(df_ohlcv: pd.DataFrame, 
                  periods: list, 
                  price_column: str = 'close') -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages (EMAs) for a list of periods.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing OHLCV data.
        periods (list): A list of periods (integers) for which to calculate EMAs.
        price_column (str): The column name to use for price data (default is 'close').
        
    Returns:
        pd.DataFrame: The DataFrame with additional EMA columns for each period.
    """
    for period in periods:
        ema_column = f'ema_{period}'
        df_ohlcv[ema_column] = df_ohlcv[price_column].ewm(span=period, adjust=False).mean()
    return df_ohlcv




def calculate_bollinger_bands(df_ohlcv: pd.DataFrame, 
                              period: int = 20, 
                              price_column: str = 'close', 
                              std_multiplier: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing OHLCV data.
        period (int): The moving average period for Bollinger Bands (default is 20).
        price_column (str): The column name to use for price data (default is 'close').
        num_std (int): The number of standard deviations for the bands (default is 2).
        
    Returns:
        pd.DataFrame: The DataFrame with added Bollinger Band columns: 'bollinger_mid',
                      'bollinger_upper', and 'bollinger_lower'.
    """
    # Calculate the rolling mean (middle band)
    df_ohlcv['bollinger_mid_period'] = df_ohlcv[price_column].rolling(window=period).mean()
    
    # Calculate the rolling standard deviation
    rolling_std = df_ohlcv[price_column].rolling(window=period).std()
    
    # Calculate the upper and lower bands
    df_ohlcv['bollinger_upper_period'] = df_ohlcv['bollinger_mid_period'] + (rolling_std * std_multiplier)
    df_ohlcv['bollinger_lower_period'] = df_ohlcv['bollinger_mid_period'] - (rolling_std * std_multiplier)
    
    return df_ohlcv



def get_data_for_emas_strategy_backtesting(df_emas: pd.DataFrame,
                                          ema_period1: int,
                                          ema_period2: int,
                                          ema_period3: int,
                                          ema_period4: int) -> pd.DataFrame:

    df_emas_with_orders = df_emas.copy()

    df_emas_with_orders['emas_optimal_position'] = ((df_emas_with_orders[f'ema_{ema_period1}'] >= df_emas_with_orders[f'ema_{ema_period2}']) * \
                                                       (df_emas_with_orders[f'ema_{ema_period2}'] >= df_emas_with_orders[f'ema_{ema_period3}']) * \
                                                       (df_emas_with_orders[f'ema_{ema_period3}'] >= df_emas_with_orders[f'ema_{ema_period4}'])).astype(int)

    df_emas_with_orders['enterLong'] = df_emas_with_orders['emas_optimal_position'].diff()

    return df_emas_with_orders





def get_emas_strategy_trades_not_vectorized(df_emas_with_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trades from a DataFrame based on the enterLong column.
    
    Args:
        df_emas_with_orders (pd.DataFrame): DataFrame containing a column 'enterLong'
                                            (1 for entry, -1 for exit, 0 for nothing).
                                            
    Returns:
        pd.DataFrame: DataFrame containing trade details (entry_time, entry_price,
                      exit_time, exit_price, and profit).
    """
    trades = []
    current_trade = {}

    # Iterate over the rows of the DataFrame
    for idx, row in df_emas_with_orders.iterrows():
        if row['enterLong'] == 1:  # Start of a trade
            current_trade = {'entry_time': row['timestamp'], 'entry_price': row['close']}
        elif row['enterLong'] == -1 and current_trade:  # End of a trade
            current_trade['exit_time'] = row['timestamp']
            current_trade['exit_price'] = row['close']
            trades.append(current_trade)
            current_trade = {}

    # Convert the list of trades to a DataFrame
    df_trades = pd.DataFrame(trades)
    df_trades['yield'] = 100 * (df_trades['exit_price'] - df_trades['entry_price']) / df_trades['entry_price']
    return df_trades



def get_emas_strategy_trades_vectorized(df_emas_with_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trades from a DataFrame based on the enterLong column.
    
    Args:
        df_emas_with_orders (pd.DataFrame): DataFrame containing a column 'enterLong'
                                            (1 for entry, -1 for exit, 0 for nothing).
                                            
    Returns:
        pd.DataFrame: DataFrame containing trade details (entry_time, entry_price,
                      exit_time, exit_price, and profit).
    """
    first_index = df_emas_with_orders[df_emas_with_orders['enterLong'] == 1].index[0]
    last_index = df_emas_with_orders[df_emas_with_orders['enterLong'] == -1].index[-1]

    df_emas_with_orders_first_last_index = df_emas_with_orders.loc[first_index: last_index]
    df_orders = df_emas_with_orders_first_last_index[df_emas_with_orders_first_last_index['enterLong'].isin([1, -1])]
    
    entries = df_orders[df_orders['enterLong'] == 1].reset_index(drop = True)
    exits = df_orders[df_orders['enterLong'] == -1].reset_index(drop = True)

    df_trades = pd.DataFrame({
        'entry_time': entries['timestamp'],
        'entry_price': entries['close'],
        'exit_time': exits['timestamp'],
        'exit_price': exits['close']
    })
    
    df_trades['yield'] = 100 * (df_trades['exit_price'] - df_trades['entry_price']) / df_trades['entry_price']
    
    return df_trades



def get_data_for_bollinger_bands_backtesting(df_emas_bollinger_bands: pd.DataFrame,
                                             ema_period1: int,
                                             ema_period2: int,
                                             ema_period3: int,
                                             ema_period4: int,
                                             bollinger_bands_period: int,
                                             bollinger_bands_std_multiplier: float,
                                             bollinger_bands_buy_column: str,
                                             bollinger_bands_sell_column: str,
                                             using_emas: bool = True) -> pd.DataFrame:

    df_emas_bollinger_bands_with_orders = calculate_bollinger_bands(df_emas_bollinger_bands, 
                                                                    period = bollinger_bands_period, 
                                                                    price_column = 'close', 
                                                                    std_multiplier = bollinger_bands_std_multiplier)
    
    enterLong_columns = []

    if using_emas:
        df_emas_bollinger_bands_with_orders['emas_optimal_position'] = ((df_emas_bollinger_bands_with_orders[f'ema_{ema_period1}'] >= df_emas_bollinger_bands_with_orders[f'ema_{ema_period2}']) * \
                                                           (df_emas_bollinger_bands_with_orders[f'ema_{ema_period2}'] >= df_emas_bollinger_bands_with_orders[f'ema_{ema_period3}']) * \
                                                           (df_emas_bollinger_bands_with_orders[f'ema_{ema_period3}'] >= df_emas_bollinger_bands_with_orders[f'ema_{ema_period4}'])).astype(int)
        enterLong_columns.append('emas_optimal_position')

    df_emas_bollinger_bands_with_orders['bollinger_bands_long_condition'] = (df_emas_bollinger_bands_with_orders['close'] <= df_emas_bollinger_bands_with_orders[bollinger_bands_buy_column]).astype(int)
    enterLong_columns.append('bollinger_bands_long_condition')
    
    df_emas_bollinger_bands_with_orders['enterLong'] = df_emas_bollinger_bands_with_orders[enterLong_columns].prod(axis = 1)
    df_emas_bollinger_bands_with_orders['exitLong'] = (df_emas_bollinger_bands_with_orders['close'] >= df_emas_bollinger_bands_with_orders[bollinger_bands_sell_column]).astype(int)
    
    df_emas_bollinger_bands_with_orders['order'] = (df_emas_bollinger_bands_with_orders['enterLong'] - df_emas_bollinger_bands_with_orders['exitLong']).replace({0: np.nan}).ffill().diff()/2
    
    return df_emas_bollinger_bands_with_orders




def get_bollinger_strategy_trades_vectorized(df_ema_bollinger_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trades from a DataFrame based on the enterLong column.
    
    Args:
        df_emas_with_orders (pd.DataFrame): DataFrame containing a column 'enterLong'
                                            (1 for entry, -1 for exit, 0 for nothing).
                                            
    Returns:
        pd.DataFrame: DataFrame containing trade details (entry_time, entry_price,
                      exit_time, exit_price, and profit).
    """
    try:
        first_index = df_ema_bollinger_orders[df_ema_bollinger_orders['order'] == 1].index[0]
        last_index = df_ema_bollinger_orders[df_ema_bollinger_orders['order'] == -1].index[-1]
    
        df_ema_bollinger_orders_first_last_index = df_ema_bollinger_orders.loc[first_index: last_index]
        df_orders = df_ema_bollinger_orders_first_last_index[df_ema_bollinger_orders_first_last_index['order'].isin([1, -1])]
        
        entries = df_orders[df_orders['order'] == 1].reset_index(drop = True)
        exits = df_orders[df_orders['order'] == -1].reset_index(drop = True)
    
        df_trades = pd.DataFrame({
            'entry_time': entries['timestamp'],
            'entry_price': entries['close'],
            'exit_time': exits['timestamp'],
            'exit_price': exits['close']
        })
        
        df_trades['yield'] = 100 * (df_trades['exit_price'] - df_trades['entry_price']) / df_trades['entry_price']
        
        return df_trades
    except:
        return None




def get_kpis(df_trades: pd.DataFrame,
             params: dict) -> pd.DataFrame:
    res = {}
    res.update(params)
    initial_capital = res['init_capital']

    try:
        n = len(df_trades)
        final_capital = initial_capital * (1 + df_trades['yield']/100).prod()
        res['nb_trades'] = n
        res['final_capital'] = final_capital
        res['PnL'] = final_capital - initial_capital
        res['win_rate'] = len(df_trades[df_trades['yield'] >= 0]) / n
        res['avg_win'] = df_trades[df_trades['yield'] >= 0]['yield'].mean()
        res['avg_loss'] = df_trades[df_trades['yield'] < 0]['yield'].mean()
        res['risk_reward'] = res['avg_win'] / abs(res['avg_loss']) if res['avg_loss'] != 0 else 0
        
    except:
        res['nb_trades'] = 0
        res['final_capital'] = initial_capital
        res['PnL'] = 0
        res['win_rate'] = 0
        res['avg_win'] = 0
        res['avg_loss'] = 0
        res['risk_reward'] = 0
        
    return res
