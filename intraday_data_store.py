import yfinance as yf
import pandas as pd
import os
from sqlalchemy import create_engine, text

# Symbols
FUTURE_SYMBOL = 'ES=F'  # Example: S&P 500 E-mini futures
FUTURE_SYMBOLS = ['ES=F', 'ZN=F', 'ZB=F', 'ZT=F', 'ZF=F', 'NQ=F', 'HG=F',  'CL=F', 'RTY=F'] #'DX-Y.NYB',
# Constants
INTERVAL = '5m'  # 1-minute frequency
PAST_DAYS = 5  # Number of days to capture
DB_FILE = 'futures_data.db'  # SQLite database file
PARQUET_DIR = 'parquet_data'  # Directory for Parquet files

# Ensure the Parquet directory exists
os.makedirs(PARQUET_DIR, exist_ok=True)

# Maximum number of variables allowed in an SQLite query (adjust as necessary)
SQLITE_MAX_VARIABLE_NUMBER = 999  # Adjust according to your SQLite version


def get_database_connection(db_file):
    """Create a connection to the SQLite database."""
    engine = create_engine(f'sqlite:///{db_file}', echo=False, future=True)
    return engine


def create_index(db_engine, table_name, column_name):
    """Create an index on the specified column of the table."""
    with db_engine.connect() as connection:
        connection.execute(
            text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name} ON {table_name} ({column_name})"))


def download_intraday_data(symbol, interval, period='60d'):
    """Download intraday data for a given symbol."""
    data = yf.download(tickers=symbol, interval=interval, period=period)
    return data


def batch_insert_data(data, table_name, db_engine, max_vars=SQLITE_MAX_VARIABLE_NUMBER):
    """Insert data into the SQLite database in batches."""
    with db_engine.connect() as connection:
        # Calculate the maximum number of rows we can insert per batch
        num_columns = len(data.columns) + 1  # +1 for the index column
        max_rows_per_batch = max_vars // num_columns

        for i in range(0, len(data), max_rows_per_batch):
            batch_data = data.iloc[i:i + max_rows_per_batch]
            batch_data.to_sql(table_name, db_engine, if_exists='append', index=True, method='multi')

    create_index(db_engine, table_name, 'Datetime')


def store_data_in_parquet(data, parquet_dir, table_name):
    """Store data in a Parquet file for faster read/write using pyarrow."""
    parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")
    data.to_parquet(parquet_file, engine='pyarrow', compression='gzip')


def load_data_from_parquet(parquet_dir, table_name):
    """Load data from a Parquet file using pyarrow."""
    parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")
    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file, engine='pyarrow')
    else:
        return pd.DataFrame()


def load_data_from_db(table_name, db_engine):
    """Load the data from the SQLite database."""
    try:
        query = text(f"SELECT * FROM {table_name}")
        return pd.read_sql(query, db_engine, index_col='Datetime', parse_dates=True)
    except Exception:  # Handle case where table does not exist
        return pd.DataFrame()


def compare_and_update_data(new_data, existing_data):
    """Compare and update the existing data if new data is better."""
    if existing_data.empty:
        return new_data

    # Combine datasets and drop duplicates
    combined_data = pd.concat([existing_data, new_data]).drop_duplicates()

    # Sort by datetime to maintain order
    combined_data = combined_data.sort_index()

    return combined_data


def update_intraday_data(symbol, interval, db_file, parquet_dir):
    """Main function to update intraday data in the database and Parquet file."""
    table_name = f"{symbol.replace('=', '_')}_{interval}"

    # Get a database connection
    db_engine = get_database_connection(db_file)

    # Download new data
    new_data = download_intraday_data(symbol, interval)

    # Load existing data from both the database and Parquet file
    existing_data_db = load_data_from_db(table_name, db_engine)
    existing_data_parquet = load_data_from_parquet(parquet_dir, table_name)

    # Combine the existing data from both sources
    existing_data = pd.concat([existing_data_db, existing_data_parquet]).drop_duplicates()

    # Compare and update with the new data
    updated_data = compare_and_update_data(new_data, existing_data)

    # Store the updated data in the database in batches to avoid the "too many SQL variables" error
    batch_insert_data(updated_data, table_name, db_engine)

    # Store the updated data in a Parquet file
    store_data_in_parquet(updated_data, parquet_dir, table_name)

    print(f"Data updated and stored in table '{table_name}' in database '{db_file}' and Parquet file '{parquet_dir}'")


if __name__ == "__main__":
    # Run the update process daily
    for sym in FUTURE_SYMBOLS:
        update_intraday_data(sym, INTERVAL, DB_FILE, PARQUET_DIR)
