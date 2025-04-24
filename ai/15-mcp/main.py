from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from discord.ext import commands
from concurrent.futures import ThreadPoolExecutor
import os
import discord
import asyncio

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.members = True
intents.presences = True

bot_ready = False
bot = commands.Bot(command_prefix="!", intents=intents)

async def mcp_run_in_background():
    await mcp.run(transport="stdio")

@bot.event
async def on_ready():
    global bot_ready
    bot_ready = True
    print(f"Bot conectado a los siguientes servidores: {[guild.name for guild in bot.guilds]}")

mcp = FastMCP("Test")

@mcp.tool()
def getOs() -> str:
    return os.uname().sysname

@mcp.tool()
async def getDiscordUsers() -> list:
    if not bot_ready:
        return "Bot is not ready yet."
    members = []
    print("Fetching members...")
    for guild in bot.guilds:
        async for member in guild.fetch_members(limit=None):
            print(f"Member: {member.name}")
            members.append(member.name)
    print(members)
    return [guild.name for guild in bot.guilds]

async def main():
    with ThreadPoolExecutor() as executor:
        executor.submit(mcp_run_in_background())
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
