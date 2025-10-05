import regex as re
import logging


def parse_telegram_message(message: str) -> dict:
    global name
    """
    Parses a single Telegram-style message string and extracts only the required data fields.
    """
    try:
        data = {}

        # Split message into lines and strip whitespace
        lines = [line.strip() for line in message.split("\n")]

        # -- REGEX PATTERNS --
        ca_pattern = re.compile(r"([A-Za-z0-9]{30,})")
        mc_pattern = re.compile(
            r"💰\s*MC:\s*\$([\d\.]+[KMB]?)\s*•\s*🔝\s*\$([\d\.]+[KMB]?)"
        )
        liq_pattern = re.compile(r"💧\s*Liq:\s*\$([\d\.]+[KMB]?)\s*\(([\d\.]+)\s*SOL\)")
        age_pattern = re.compile(r"🕒\s*Age:\s*([\w\d]+)")
        snipers_pattern = re.compile(r"🔫\s*Snipers:\s*(\d+)")
        dev_pattern = re.compile(
            r"🛠️\s*Dev:\s*([\d\.]+)\s*SOL\s*\|\s*([\d\.]+)%\s*\$(\S+)"
        )

        name = None

        # --- Extract Data from Message ---
        for line in lines:
            # First line
            m = ca_pattern.search(line)
            if m:
                name = m.group(1)

            # Market Cap
            m = mc_pattern.search(line)
            if m:
                data["mc"] = convert_to_numeric(m.group(1))
                data["mc_ath"] = convert_to_numeric(m.group(2))
            elif "mc" not in data and "mc_ath" not in data:
                data["mc"] = 0
                data["mc_ath"] = 0

            # Liquidity (SOL)
            m = liq_pattern.search(line)
            if m:
                data["liq_sol"] = float(m.group(2))
            elif "liq_sol" not in data:
                data["liq_sol"] = 0

            # Age in Minutes
            m = age_pattern.search(line)
            if m:
                data["age_minutes"] = convert_age_to_minutes(m.group(1))
            elif "age_minutes" not in data:
                data["age_minutes"] = 0

            # Snipers Count
            m = snipers_pattern.search(line)
            if m:
                data["total_snipers_count"] = int(m.group(1))
            elif "total_snipers_count" not in data:
                data["total_snipers_count"] = 0

            # Dev Info
            m = dev_pattern.search(line)
            if m:
                data["dev_sol"] = float(m.group(1))
                data["dev_ror"] = float(m.group(2))
            elif "dev_sol" not in data and "dev_ror" not in data:
                data["dev_sol"] = 0
                data["dev_ror"] = 0

        # --- Parse Wallet-Type Emojis ---
        wallet_counts = {"shrimp": 0, "fish": 0, "whale": 0, "fresh": 0}
        for line in lines:
            wallet_counts["shrimp"] += line.count("🍤")
            wallet_counts["fish"] += line.count("🐟")
            wallet_counts["whale"] += line.count("🐳")
            wallet_counts["fresh"] += line.count("🌱")

        data.update(wallet_counts)
        print("Raw message:", message)
        print("Parsed data:", data, "Name:", name)

        if name is not None and data["mc"] != 0:
            return data, name
        else:
            return {}, ""

    except Exception as e:
        logging.exception(f"Error in parse_telegram_message: {e}")
        return {}


# --- Helper Functions ---
def convert_to_numeric(value):
    """Convert values like 119.9K, 1.2M to numeric values."""
    try:
        if value.endswith("K"):
            return float(value[:-1]) * 1_000
        elif value.endswith("M"):
            return float(value[:-1]) * 1_000_000
        elif value.endswith("B"):
            return float(value[:-1]) * 1_000_000_000
        return float(value) if value.replace(".", "", 1).isdigit() else 0
    except Exception:
        logging.exception("Error converting value to numeric")
        return 0


def convert_age_to_minutes(age_str):
    """Convert age formats (e.g., 4h, 2d, 1w) into minutes."""
    try:
        if "m" in age_str:
            return int(age_str.replace("m", ""))
        elif "h" in age_str:
            return int(age_str.replace("h", "")) * 60
        elif "d" in age_str:
            return int(age_str.replace("d", "")) * 1440
        elif "w" in age_str:
            return int(age_str.replace("w", "")) * 10080
        elif "mo" in age_str:
            return int(age_str.replace("mo", "")) * 43200
        return 0
    except Exception:
        logging.exception("Error converting age string to minutes")
        return 0
