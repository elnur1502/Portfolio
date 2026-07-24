from typing import Optional
from langchain.tools import tool
import chromadb
import os
from agents_core.models import embed_model
import numpy as np
import ast
import uuid
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import sys
import io
from markitdown import MarkItDown
from pydantic import BaseModel
from typing import List, Dict
from agents_core.logger import logger
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

SHORT_MEMORY = {} # only for one user, miltiple users can override short memory

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(CURRENT_DIR, "..", "data", "outputs")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
top_n_statements = 10 # Candidate pool size for user profile retrieval.

# ---------------------------------------------------------------------------
# Persistent ChromaDB client
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
Chroma_DB_DIR = os.path.join(CURRENT_DIR, "..", "chroma_db")
client = chromadb.PersistentClient(path=Chroma_DB_DIR)

profiles_collection = client.get_or_create_collection(name="preferences")


# ===========================================================================
# Code execution
# ===========================================================================

def run_code(code_python: str):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    try:
        exec(code_python, SHORT_MEMORY)
        output = redirected_output.getvalue()
        return {
            "status": "success",
            "output": str(output)
        }
    except Exception:
        return {
            "status": "error exception",
            "output": str(traceback.format_exc())
        }
    finally:
        sys.stdout = old_stdout

def reset():
        SHORT_MEMORY.clear() ## for new iterations


@tool
async def run_python(code: str) -> dict:
    """
    Execute a snippet of Python in the isolated sandbox backend.

    Use this tool when the agent needs to compute something locally that it
    cannot derive from documents (e.g. aggregations, transformations, plotting
    data, regex parsing). Forbidden modules are 'os', 'subprocess', 'socket', 'requests', 'webbrowser', 'pickle', 'sys'.

    Args:
        code: A complete, self-contained Python program. Main imports already 
              available in the sandbox image;

    Returns:
        Parsed JSON response from the sandbox as a `dict`. Typical keys:
            `{"status": "success", "output": "sys.stdout result"}`.
        On code execution failure returns a dict `{"status": "error exception", "output": "error message"}`.
        On transport or json failure returns a dict `{"status": "error json","output": "error message"}`,
        On AST parsing failure returns a dict `{"status": "error AST parsing","output": "error message"}`,
    """
    forbidden_modules = {'subprocess', 'socket', 'requests', 'webbrowser', 'pickle', 'sys'}
    try:
        clean_code = code.split('python')[1].replace('```', '')
    except:
        clean_code = code.replace('```', '')
    try:
        parsed = ast.parse(clean_code)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module in forbidden_modules:
                        return {"status": "error", "output": "You are not allowed to use command: " + str(alias.name)}
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in forbidden_modules:
                        return {"status": "error", "output": "You are not allowed to use command: " + str(node.module)}
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'exec', 'eval'}:
                        return {"status": "error", "output": "You are not allowed to use command: " + str(node.func.id)}

        try:
            r = run_code(clean_code)
            return r
        
        except ValueError as e:
            return {"status": "error json","output": f"run_python non-JSON response: {e}"}
        
    except Exception as e:
        return {"status": "error AST parsing","output": 'Problem with AST parsing, error: ' + str(e)}


# ===========================================================================
# Local document readers
# ===========================================================================

@tool
async def file_reader(path: str) -> str:
    """
    Extract plain text from a SINGLE file (not a folder) on the local filesystem.
    Layout, tables, and images are NOT preserved.
    """
    try:
        correct_path = path.replace('/app/', '').replace('app/', '')

        if not (correct_path.startswith('data/')):
            return "Error: path must be under the '/app/data/' directory."

        # ↓↓↓ вот эта проверка отсутствует — именно она и есть корень проблемы
        if not os.path.isfile(correct_path):
            if os.path.isdir(correct_path):
                return ("Error: path is a directory, not a file. "
                        "file_reader reads exactly ONE file at a time. "
                        "List the directory contents yourself and call file_reader "
                        "once per file.")
            return f"Error: file not found or not readable: {path}"

        md = MarkItDown()
        if correct_path.endswith('.ts'):
            with open(correct_path, "rb") as f:
                result = md.convert_stream(f, file_extension=".txt")
            return result.text_content
        else:
            result = md.convert(correct_path)
            return result.text_content

    except Exception as e:
        return f"Error: file_reader failed: {type(e).__name__}: {e}"


# ===========================================================================
# Retrieval helpers
# ===========================================================================

@tool
async def RAG_statements(query_text: str) -> list[str]:
    """Retrieves facts, statements, and user profile data previously memorized.

    This function queries a RAG system containing not only user preferences 
    (e.g., timezone, preferred format) but also domain-specific knowledge, 
    recurring stakeholders, and important context injected by the supervisor.

    Args:
        query_text: The search query to look up (e.g., "user's preferred chart library" 
            or "what client_id means").
        top_n_statements: The maximum number of relevant statement strings to return.

    Returns:
        A list of up to `top_n_statements` relevant profile or domain-fact strings. 
        Returns an empty list if no matches are found.
    """
    try:
        query_embedding = await embed_model.aembed_query(query_text)
        query_embed = np.array(
            [query_embedding], dtype="float32"
        )
        results = profiles_collection.query(
            query_embeddings=query_embed,
            n_results=top_n_statements,
        )
        docs = results.get("documents", [[]])[0]
        return [str(d) for d in docs]
    except Exception as e:
        return [f"Error in RAG_profile: {e}"]
    
# ===========================================================================
# Long-term memory
# ===========================================================================

class ChromaData(BaseModel):
    documents: List[str]
    metadatas: List[Dict]

@tool
async def summarization(chromadb_data: ChromaData):
    """
    Persist a single record at a time into supervisor-managed long-term memory.

    Stores one piece of information about the user — preferences, recurring
    facts, habits, explicit notes, or any other annotation worth remembering
    across sessions — into the ChromaDB ``profiles_collection``.

    Use this whenever the supervisor concludes that something should be
    retained as part of the user profile. **Only the supervisor agent is
    allowed to call this** — other agents must route through the supervisor
    instead of invoking it directly.

    Parameters
    ----------
    chromadb_data : dict
        Payload holding exactly one record:
        - ``documents`` (list[str]): A single-element list with the text
          to embed and store.
        - ``metadatas`` (list[dict]): A single-element list with the
          metadata for that document (e.g. ``source``, ``timestamp``,
          ``user_id``).

    Returns
    -------
    str
        ``"Successfully added to the long-term memory"`` on success,
        or ``"Could not add to the long-term memory, error: <msg>"``
        when the persistence step fails.
    """
    try:
        id_number = [str(uuid.uuid4())]
        docs = chromadb_data.documents
        meta = chromadb_data.metadatas
        doc_embedding = await embed_model.aembed_query(docs[0])
        docs_embed = np.array(
            [doc_embedding], dtype="float32"
        )

        profiles_collection.add(ids=id_number, documents=docs, metadatas=meta, embeddings=docs_embed)
        return 'Successfully added to the long-term memory'
    except Exception as e:
        return "Could not add to the long-term memory, error: " + str(e)

# ===========================================================================
# Playwright MCP
# ===========================================================================

class PersistentPlaywright:
    def __init__(self):
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools: dict[str, object] | None = None

    async def start(self):
        if self._session is not None:
            return
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(StdioServerParameters(
                command="npx",
                args=["@playwright/mcp@latest", "--isolated"],
            ))
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        tools = await load_mcp_tools(self._session)
        self._tools = {t.name: t for t in tools}
        logger.info("Playwright MCP started")

    async def close(self):
        if self._stack:
            await self._stack.aclose()
            self._stack = self._session = self._tools = None
            logger.info("Playwright MCP closed")

    async def playwright_tools(self) -> list:
        await self.start()
        return [t for t in self._tools.values() if t.name.startswith("browser_")]

    async def call(self, action: str, **kwargs):
        await self.start()
        target = (
            self._tools.get(f"browser_{action}")
            or self._tools.get(action)
            or next((t for t in self._tools.values()
                     if t.name.endswith(f"_{action}")), None)
        )
        if not target:
            logger.info("Playwright MCP action %r not found. Available: %s", action, sorted(self._tools),)
            raise RuntimeError(f"action {action!r} not found")
        logger.info("MCP call: %s args=%s", target.name, sorted(kwargs))
        t0 = time.perf_counter()
        try:
            result = await target.ainvoke(kwargs)
        except Exception as e:
            logger.info("MCP call %s failed: %s", target.name, e)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("MCP %s done in %.0fms", target.name, elapsed_ms)
        return result


pw = PersistentPlaywright()


async def get_playwright():
    await pw.start()
    #await pw.call("snapshot")


async def playwright_tools():
    return await pw.playwright_tools()


# ===========================================================================
# Run tests
# ===========================================================================

@tool
async def run_tests(test_path: Optional[str] = None) -> str:
    """Execute a Playwright test suite and return its output.

    Runs ``npx playwright test`` against the given path. If ``test_path`` is
    omitted, the entire suite in the workspace is executed. The path can
    point to a single ``.spec.ts``/``.spec.js`` file or to a directory
    containing specs.

    Args:
        test_path: Workspace-relative path to a test file or directory of
            specs. If ``None``, all Playwright tests in the workspace are run.

    Returns:
        str: The raw ``stdout`` of a successful run. If Playwright exits
        non-zero or the subprocess fails to start, an ``"Error: ..."``
        string containing ``stderr`` (or the exception message) is
        returned instead.
    """
    npx_executable = "npx.cmd" if sys.platform == "win32" else "npx"
    cmd = [npx_executable, "playwright", "test"]
    
    if test_path:
        correct_path = test_path.replace('/app/', '').replace('app/', '')
        try:
            correct_path = correct_path.split('outputs/')[1]
        except IndexError:
            pass
            
        if correct_path.startswith('tests/'):
            cmd.append(correct_path)
        else:
            return "Error: test_path must be under the 'tests/' directory"

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=TESTS_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            
        )
        
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        
        # Playwright при падении тестов возвращает не 0, 
        # но нам всё равно нужен stdout, чтобы увидеть, какие тесты упали
        output = stdout.decode("utf-8", errors="replace")
        errors = stderr.decode("utf-8", errors="replace")
        
        if proc.returncode == 0:
            return output
        return f"Error (code {proc.returncode}):\n{output}\n{errors}"
        
    except Exception as e:
        return f"Execution Exception: {str(e)}"
