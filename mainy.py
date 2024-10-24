import pandas as pd
import yfinance as yf
from datetime import datetime


# def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
def fetch_data(symbol: str, period: str) -> pd.DataFrame:

    # Fetch historical stock data
    tkr = yf.Ticker(symbol)
    # data = tkr.history(start=start_date, end=end_date)
    data = tkr.history(period=period)

    data.reset_index(inplace=True)

    # Calculate EMAs
    data['EMA_21High'] = data['High'].ewm(span=21, adjust=False).mean()
    data['EMA_21Low'] = data['Low'].ewm(span=21, adjust=False).mean()
    data['EMA_200Close'] = data['Close'].ewm(span=200, adjust=False).mean()

    return data


def simulate_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['in_position'] = False
    df['entry_price'] = 0.0
    df['exit_price'] = 0.0
    df['buy_signal'] = False
    df['sell_signal'] = False
    df['profit_loss'] = 0.0
    df['last_entry_price'] = 0.0
    df['actual_capital'] = 0.0
    df['inventory'] = 10000.0  # Starting cash

    # Ensure 'Date' column is in datetime64[ns] format without timezone
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    df['actual_buy_date'] = pd.NaT
    df['actual_sell_date'] = pd.NaT

    alr_bought = False

    for i in range(1, len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        try:
            # Buy Logic
            if (not df.iloc[i - 1]['in_position'] and
                current_row['Low'] > current_row['EMA_21High'] and
                current_row['Close'] > current_row['Open'] and
                current_row['EMA_21Low'] > current_row['EMA_200Close'] and
                    not alr_bought):

                if next_row['High'] > current_row['High']:
                    entry_price = current_row['High']
                    current_inventory = df['inventory'].iloc[i]

                    shares_to_buy = int(
                        min(current_inventory, 10000.0) // entry_price)

                    if shares_to_buy > 0:
                        total_cost = shares_to_buy * entry_price

                        df.loc[i + 1:, 'in_position'] = True
                        df.loc[i + 1, 'entry_price'] = entry_price
                        df.loc[i + 1, 'buy_signal'] = True
                        df.loc[i + 1:, 'last_entry_price'] = entry_price

                        df.loc[i + 1:, 'actual_capital'] = shares_to_buy
                        df.loc[i + 1:, 'inventory'] = df.loc[i,
                                                             'inventory'] - total_cost

                        # Store actual buy date
                        df.loc[i + 1, 'actual_buy_date'] = df.loc[i, 'Date']

                        alr_bought = True

            # Sell Logic
            if (current_row['Open'] > current_row['Close'] and
                current_row['in_position'] and
                current_row['Close'] < current_row['EMA_21Low'] and
                    alr_bought):

                if next_row['entry_price'] < current_row['Low']:
                    exit_price = current_row['Low']
                    shares_held = df['actual_capital'].iloc[i]

                    total_proceeds = shares_held * exit_price

                    df.iloc[i + 1, df.columns.get_loc('sell_signal')] = True
                    df.iloc[i + 1,
                            df.columns.get_loc('exit_price')] = exit_price
                    df.loc[i + 1:, 'in_position'] = False
                    last_entry = df['last_entry_price'].iloc[i]
                    if last_entry > 0:
                        profit_loss = (exit_price - last_entry) * shares_held
                        df.iloc[i + 1,
                                df.columns.get_loc('profit_loss')] = profit_loss

                        df.loc[i + 1:, 'inventory'] = df.loc[i,
                                                             'inventory'] + total_proceeds
                        df['buy_sell_date'] = df['Date'].iloc[i + 1]

                        df.loc[i + 1:, 'actual_capital'] = 0.0
                        df.loc[i + 1:, 'last_entry_price'] = 0.0

                        # Store actual sell date
                        df.loc[i + 1, 'actual_sell_date'] = df.loc[i, 'Date']

                        alr_bought = False

        except IndexError:
            print(f"Skipping stock due to IndexError at index {
                  i} for this stock.")
            break

    return df


def main():
    """
    Main function to execute the trading strategy and save the results.
    """
    main_df = pd.DataFrame()

    stock_data = [
        'RELIANCE.NS', 'LT.NS', 'ABB.NS', 'AMBER.NS',
        'INFY.NS',    # Infosys Limited
        'TCS.NS'  # Tata Consultancy Services
        # 'HDFC.NS',    # HDFC Bank
        # 'ICICIBANK.NS',  # ICICI Bank
        # 'SBIN.NS',    # State Bank of India
        # 'BHARTIARTL.NS',  # Bharti Airtel
        # 'ASIANPAINT.NS',  # Asian Paints
        # 'BAJAJ-AUTO.NS',  # Bajaj Auto
        # 'HEROMOTOCO.NS',  # Hero MotoCorp
        # 'MARUTI.NS',  # Maruti Suzuki
        # 'TITAN.NS',   # Titan Company
        # 'ULTRACEMCO.NS',  # UltraTech Cement
        # 'WIPRO.NS',   # Wipro Limited
        # 'HINDUNILVR.NS'  # Hindustan Unilever
    ]
    skipped_stocks = []
    for stock in stock_data:
        try:
            # data = fetch_data(f"{stock}", "2000-01-01", "2024-10-15")
            data = fetch_data(f"{stock}", "max")

            result_df = simulate_strategy(data)

            # Get initial and final prices
            initial_price = data.iloc[0]['Close']
            final_price = data.iloc[-1]['Close']

            # starting date and ending date
            stock_start_date = data.iloc[0]['Date']
            stock_end_date = data.iloc[-1]['Date']

            # Total profit/loss for the stock
            total_profit_loss = result_df['profit_loss'].sum()

            # Final inventory value
            final_inventory = result_df['inventory'].iloc[-1]

            # Total return = Final Inventory - Initial Investment (10000)
            total_return = final_inventory - 10000.0

            print(f"\nInitial Investment: 10000.00")
            print(f"Final Inventory for {stock}: {final_inventory:.2f}")
            print(f"Total Return for {stock}: {total_return:.2f}")
            print(f"Initial Price of {stock}: {initial_price:.2f}")
            print(f"Final Price of {stock}: {final_price:.2f}")

            # Get the last 3 trades (buy/sell signals)
            last_3 = result_df[result_df['buy_signal']
                               | result_df['sell_signal']].tail(1)

            # Add extra columns for main_df
            last_3['stock_name'] = stock
            last_3['last_buy_sell_date'] = result_df['buy_sell_date']
            last_3['total_profit_loss'] = total_profit_loss
            last_3['initial_price'] = initial_price
            last_3['final_price'] = final_price
            last_3['total_difference_stock'] = final_price - initial_price
            last_3['stock_start_date'] = stock_start_date
            last_3['stock_end_date'] = stock_end_date

            # Define the two date strings
            date1 = stock_start_date
            date2 = stock_end_date

            # Convert the date strings into datetime objects
            fmt = "%Y-%m-%d %H:%M:%S%z"
            datetime1 = datetime.strptime(str(date1), fmt)
            datetime2 = datetime.strptime(str(date2), fmt)

            # Subtract the dates to get the difference
            date_difference = datetime2 - datetime1

            # Extract the number of days
            days_difference = date_difference.days

            last_3['total_days'] = days_difference

            # Concatenate the last 3 rows to the main_df
            main_df = pd.concat([main_df, last_3], axis=0)

            # Trading stats
            num_trades = len(result_df[result_df['buy_signal']])
            profitable_trades = result_df[result_df['profit_loss'] > 0]
            win_rate = len(profitable_trades) / num_trades * \
                100 if num_trades > 0 else 0
            avg_profit = profitable_trades['profit_loss'].mean() if len(
                profitable_trades) > 0 else 0
            avg_loss = result_df[result_df['profit_loss']
                                 < 0]['profit_loss'].mean()

            print(f"\nNumber of trades: {num_trades}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Average profit per winning trade: {avg_profit:.2f}")
            print(f"Average loss per losing trade: {avg_loss:.2f}")

        # CATCH POTENTIAL INDEXERROR WHILE FETCHING DATA FOR THE STOCK
        except IndexError:
            print(f"Skipping stock {
                  stock} due to IndexError while fetching data or simulating strategy.")
            skipped_stocks.append(stock)

    # Save the results to CSV
    main_df[['stock_name', 'stock_start_date', 'stock_end_date', 'total_days', 'buy_signal', 'sell_signal', 'last_buy_sell_date', 'actual_buy_date', 'actual_sell_date', 'total_profit_loss',
             'initial_price', 'final_price', 'total_difference_stock']].to_csv("v4_max_period_with_total_days.csv", index=False)
    print(skipped_stocks)


if __name__ == "__main__":
    main()
