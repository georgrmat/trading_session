import pandas as pd

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


