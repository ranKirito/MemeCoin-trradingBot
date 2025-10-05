import sqlite3 as sq


def initialize_database(db_name: str):
    """
    Initialize the SQLite database and create the telegram_data table
    with only the relevant columns used by the AI model.
    """
    conn = sq.connect(db_name)
    cursor = conn.cursor()

    # Create a table with the required columns
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS telegram_data (
            address TEXT,
            mc REAL,
            mc_ath REAL,
            liq_sol REAL,
            age_minutes INTEGER,
            total_snipers_count INTEGER,
            dev_sol REAL,
            dev_ror REAL,
            shrimp INTEGER,
            fish INTEGER,
            whale INTEGER,
            fresh INTEGER,
            multiplier INTEGER
        )
        """
    )

    # Commit changes and close connection
    conn.commit()
    conn.close()


initialize_database("telegram_data.db")


def get_connection():
    """Establish a connection to the SQLite database."""
    return sq.connect("telegram_data.db")


async def insert_data(data, name=""):
    """
    Insert a record into the telegram_data table.

    Parameters:
        data: Dictionary containing the row values to insert.
    """
    conn = get_connection()
    sql = """
    INSERT INTO telegram_data (
        address, mc, mc_ath, liq_sol, age_minutes, total_snipers_count, 
        dev_sol, dev_ror, shrimp, fish, whale, fresh, multiplier
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        data_tuple = (
            name,
            data.get("mc", 0.0),
            data.get("mc_ath", 0.0),
            data.get("liq_sol", 0.0),
            data.get("age_minutes", 0),
            data.get("total_snipers_count", 0),
            data.get("dev_sol", 0.0),
            data.get("dev_ror", 0.0),
            data.get("shrimp", 0),
            data.get("fish", 0),
            data.get("whale", 0),
            data.get("fresh", 0),
            0.0,
        )
        conn.cursor().execute(sql, data_tuple)
        conn.commit()
    except sq.Error as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()


async def update_multiplier(name, multiplier):
    """
    Update the multiplier column for a given token name (address).

    Parameters:
        name (str): The contract address of the token.
        multiplier (float): The new multiplier value.
    """
    conn = get_connection()
    sql = """
    UPDATE telegram_data
    SET multiplier = ?
    WHERE address = ?
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (multiplier, name))
        conn.commit()
    except sq.Error as e:
        print(
            f"Error updating multiplier: {e}, more so the coin was probably not in the database"
        )
    finally:
        conn.close()
