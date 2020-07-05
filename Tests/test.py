
from Bot.Bot import Bot, conversation


def main():
    bot = Bot(updating_model= False, epochs=2000)
    conversation(bot)


if __name__ == '__main__':
    main()