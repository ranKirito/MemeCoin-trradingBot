import telethon
from urllib.parse import urlparse, parse_qs


async def send_message(client, chat_id, message):
    await client.send_message(chat_id, f"/start {message}")


async def parse_message(message, choice) -> str:
    # parse if message is from New Trending
    if choice == 1:
        ca = parse_qs(urlparse(message.reply_markup.rows[0].buttons[1].url).query)[
            "start"
        ][0]
    # parse if message is from is up
    else:
        ca = urlparse(message.entities[0].url).path.strip("/").split("/")[-1]
    return ca


async def get_multiplier(message) -> float:
    iterator = iter(message)
    for i in iterator:
        if i == "up":
            the_amount_it_went_up = next(iterator, None)

            if "%" in the_amount_it_went_up:
                break
            try:
                multi = int(the_amount_it_went_up[0:-1])
            except:
                multi = float(the_amount_it_went_up[0:-1])
            return multi

    return 0
