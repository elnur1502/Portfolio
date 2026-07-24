import uuid
import asyncio
import os
import re
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from telegramify_markdown import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents_core.tools import reset
from agents.supervisor import build_supervisor
from agents_core.tools import pw, get_playwright
from agents_core.logger import logger
import subprocess


try:
    project_langfuse = Langfuse(public_key=os.getenv('LANGFUSE_PUBLIC_KEY'), secret_key=os.getenv('LANGFUSE_SECRET_KEY'), host=os.getenv('LANGFUSE_BASE_URL'))
    langfuse_handler = CallbackHandler(public_key=os.getenv('LANGFUSE_PUBLIC_KEY'))
    logger.info('Langfuse init success')
except Exception as e:
    project_langfuse = None
    langfuse_handler = None
    logger.info(f"Langfuse init failed, running without observability: {e}")

rec_limit = int(os.getenv('rec_limit'))
user_filter = filters.User(user_id=int(os.getenv("telegram_user_id")))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)

graph_supervisor = None

active_streams: dict[str, asyncio.Task] = {}


def make_config() -> dict:
    callbacks = [cb for cb in [langfuse_handler] if cb is not None]
    return {
        "configurable": {"thread_id": "user_session_" + str(uuid.uuid4())},
        "recursion_limit": rec_limit,
        "callbacks": callbacks,
    }


config = make_config()


async def run_graph(chat_id: str, text: str, update: Update):
    task_text = [HumanMessage(content=text)]
    stream = graph_supervisor.astream(
        {"messages": task_text},
        stream_mode="updates",
        config=config,
    )
    try:
        async for chunk in stream:
            for _node_name, node_output in chunk.items():
                if "messages" in node_output:
                    new_message = node_output["messages"][-1]
                    if isinstance(new_message, AIMessage) and new_message.content:
                        clean_content = re.sub(
                            r"<think>.*?</think>", "", new_message.content, flags=re.DOTALL
                        ).strip()
                        try:
                            text_to_send = markdownify(clean_content)
                        except Exception as e:
                            text_to_send = clean_content
                            logger.info('Problem with response makrdownifying. Error: ' + str(e))
                        for chunk_text in text_splitter.split_text(text_to_send):
                            await update.message.reply_text(
                                chunk_text, parse_mode=ParseMode.MARKDOWN_V2
                            )

        graph_state = graph_supervisor.get_state(config)
        if graph_state.next:
            last_message = graph_state.values["messages"][-1]
            if last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    if tool_call["name"] == "call_human":
                        try:
                            text_to_send = markdownify(tool_call["args"].get("query", ""))
                        except Exception as e:
                            text_to_send = clean_content
                            logger.info('Problem with response makrdownifying. Error: ' + str(e))
                        for chunk_text in text_splitter.split_text(text_to_send):
                            await update.message.reply_text(
                                chunk_text, parse_mode=ParseMode.MARKDOWN_V2
                            )
    except asyncio.CancelledError:
        raise
    finally:
        active_streams.pop(chat_id, None)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global config
    chat_id = config["configurable"]["thread_id"]

    if update.message.text == "/new":
        if chat_id in active_streams:
            active_streams[chat_id].cancel()
            try:
                await active_streams[chat_id]
            except (asyncio.CancelledError, Exception):
                pass
        active_streams.pop(chat_id, None)

        config = make_config()
        try:
            reset()
            logger.info('Short-memory successfully reseted.')
        except Exception:
            await update.message.reply_text(
                "Couldn't reset short-term memory in sandbox container. Continue without reseting short-memory."
            )
            logger.info("Couldn't reset short-term memory in sandbox container. Continue without reseting short-memory.")
        await update.message.reply_text("New session started. What's your next task?")
        logger.info("New session started.")
        return

    if not update.message or not update.message.text:
        return

    if update.message.text == "/interrupt":
        if chat_id in active_streams:
            task = active_streams.pop(chat_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.info(f"Error waiting for canceled task: {e}")
                print(f"Error waiting for canceled task: {e}")
            await update.message.reply_text(
                "The graph has been stopped. The next message will be processed as a correction/clarification."
            )
            logger.info("The graph has been stopped. The next message will be processed as a correction/clarification.")
        else:
            await update.message.reply_text("Nothing to stop.")
            logger.info("Nothing to stop")
        return

    if chat_id in active_streams:
        await update.message.reply_text(
            "The previous task is still running. Firstly /interrupt, then a new message."
        )
        logger.info("The previous task is still running. Firstly /interrupt, then a new message.")
        return

    task = asyncio.create_task(run_graph(chat_id, update.message.text, update))
    active_streams[chat_id] = task


async def main():
    global graph_supervisor

    try:
        subprocess.run("npx playwright clear-cache", shell=True, check=True)
    except subprocess.CalledProcessError:
        logger.info("Could not clear Playwright cache. Continuing...")

    await get_playwright()

    graph_supervisor = await build_supervisor()

    app = Application.builder().token(os.getenv("telegram_bot_api")).build()
    app.add_handler(MessageHandler(user_filter & filters.TEXT, handle_message))

    async with app:
        await app.start()
        await app.updater.start_polling()
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass

    await pw.close()


if __name__ == "__main__":
    asyncio.run(main())
