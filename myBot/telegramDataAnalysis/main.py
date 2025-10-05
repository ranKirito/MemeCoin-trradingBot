# main.py
from telethon import TelegramClient, events
from dotenv import load_dotenv
import asyncio
import telegram_helper_functions as helper
import os
from new_parser import parse_telegram_message
import database_helper_functions as dbh

load_dotenv()

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
group_id1 = os.getenv("GROUP1")
group_id2 = os.getenv("GROUP2")
client = TelegramClient("anon", api_id, api_hash)


# Telegram event handlers
@client.on(events.NewMessage(chats=group_id1))
async def handler_group1(event):
    message = event.message.message
    if "is up" in message:
        if hasattr(event.message, "entities") and hasattr(
            event.message.entities[0], "url"
        ):
            asyncio.create_task(process_coin_up(event))


@client.on(events.NewMessage(chats=group_id2))
async def handler_group2(event):
    asyncio.create_task(process_database_insertion(event))


async def process_coin_up(event):
    """
    Process messages indicating a coin has gone up.
    """
    try:
        ca = await helper.parse_message(event.message, 2)
        multi = await helper.get_multiplier(event.message.message.split(" "))
        await dbh.update_multiplier(ca, multi)
        print(f"A coin {ca} went up by {multi}x")
    except Exception as e:
        print(f"Error in process_coin_up: {e}")


async def process_database_insertion(event):
    """
    Parse the Telegram message and insert data into the database.
    """
    try:
        parsed_data, name = parse_telegram_message(event.message.message)
        for k, v in parsed_data.items():
            print(f"{k}: {v}")
        await dbh.insert_data(parsed_data, name)  # Make sure insert_data is async
    except Exception as e:
        print(f"Error in process_database_insertion: {e}")


# Main entry point
async def main():
    print("Starting Telegram scraper...")
    await client.start()
    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped manually.")
