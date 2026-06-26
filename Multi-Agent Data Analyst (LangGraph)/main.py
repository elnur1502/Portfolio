from telegram import Update
from telegram.ext import (Application, ContextTypes, MessageHandler, filters)

from graph import agent
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import asyncio
import time
from agents.agent_logs import logger

load_dotenv()

rec_limit = 100

#only selected user can use this agent
user_filter = filters.User(user_id=int(os.getenv("telegram_user_id")))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    task_text = [HumanMessage(content=update.message.text)]
    logger.info(50*'*' + '\nNEW TASK:\n' + update.message.text)

    await update.message.reply_text(f'Working on it...')
    start_time = time.time()
    result = await asyncio.to_thread(
        agent.invoke,
        {"messages": task_text},
        {"recursion_limit": rec_limit}
    )
    end_time = time.time()
    duration = end_time - start_time

    work_done = result["end_of_steps"]
    if work_done == True:
        file_path = result["required_output"]
        await update.message.reply_text(
        f"Done. Check the attached file\n\nTime spent: {duration:.2f}s"
        )

        with open(file_path, "rb") as f:
            await update.message.reply_document(document=f)
            f.close()

    else:
        await update.message.reply_text(
            f"Something went wrong, please check logs\n\nTime spent: {duration:.2f}s"
        )


def main():
    app = Application.builder().token(os.getenv("telegram_bot_api")).build()

    app.add_handler(MessageHandler(user_filter & filters.TEXT, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
