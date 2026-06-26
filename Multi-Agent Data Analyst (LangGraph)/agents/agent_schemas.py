from typing import Literal, Optional
from pydantic import BaseModel, Field


# ===================== PLANNING =====================

class Plan_steps(BaseModel):
    """A single atomic step in the execution plan.
    Each step must be independent and self-contained: the executor model
    sees only this step's fields, plus the list of variables produced so far.
    """

    step_id: int = Field(
        description=(
            "Sequential step number starting from 1. "
            "Steps MUST be numbered 1, 2, 3... without gaps. "
            "Use this as the canonical reference when later steps depend on prior outputs."
        ),
        ge=1,
    )

    action: Literal["python_interpreter", "sql_query", "save_to_file"] = Field(
        description=(
            "Type of operation to execute at this step. "
            "Choose exactly one:\n"
            "- 'sql_query': data extraction, filtering, or simple aggregation from the database. "
            "Use for any SELECT-only operation. SQL output becomes a DataFrame variable in the next step.\n"
            "- 'python_interpreter': data transformation, complex aggregation, cleaning, joining, "
            "reshaping. Reads variables produced by previous steps and produces a new variable.\n"
            "- 'save_to_file': MUST be used ONLY for the final step. Writes a variable to a file "
            "in the 'data/' folder. The executor handles file I/O for you — do NOT use pandas "
            ".to_csv/.to_excel inside a python_interpreter step."
        ),
    )

    description: str = Field(
        description=(
            "One-sentence summary of WHAT this step achieves from the user's perspective. "
            "Example: 'Aggregate total sales per region'. "
            "Keep it short (under 15 words) — the detailed 'how' goes into 'instructions'."
        ),
    )

    input: str = Field(
        description=(
            "Explicit list of inputs this step needs, comma-separated. "
            "Examples:\n"
            "- 'sales_data.csv' (raw file for the first step)\n"
            "- 'df_sales' (a DataFrame variable produced by a previous step — use the EXACT name)\n"
            "- 'orders_table, clients_table' (multiple SQL tables)\n"
            "NEVER use vague references like 'previous result', 'the data', 'above output'. "
            "If no inputs are needed (e.g., first step reading a file or table by name), "
            "write 'none' or list the file/table name explicitly."
        ),
    )

    output: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED for 'python_interpreter' AND 'save_to_file' actions; leave null "
            "only for pure intermediate 'sql_query' steps where the result flows directly "
            "into the next python step. "
            "Naming convention: prefix 'df_' for DataFrames, 'file_' for final file outputs "
            "(must include 'data/' folder, e.g. 'data/file_report'). "
            "Use 'result_' for scalar/non-DataFrame outputs. "
            "This name is how ALL later steps will reference this step's result — keep it "
            "stable and descriptive."
        ),
    )

    file_name: Optional[str] = Field(
        default=None,
        description=(
            "Filename with extension for the FINAL output file. "
            "Set ONLY when action='save_to_file'. Format examples: "
            "'report.xlsx' (default), 'report.csv' (only if user requested CSV), "
            "'report.json' (only if user requested JSON). "
            "Default extension when user didn't specify format is .xlsx."
        ),
    )

    instructions: str = Field(
        description=(
            "Detailed step-by-step instructions for the executor model. "
            "This is the MOST IMPORTANT field — the executor follows it literally. "
            "Must include:\n"
            "1. The exact operation to perform (e.g. 'Group by region and sum amount').\n"
            "2. Exact column/table names from the data.\n"
            "3. Validation check: 'Verify the result DataFrame is not empty and contains "
            "columns X, Y, Z'. Always include at least one validation.\n"
            "4. For 'save_to_file' steps: explicitly state the file format and that the "
            "executor should use the save_to_file action (not pandas to_csv/to_excel).\n"
            "Do NOT include any executable code — only natural-language instructions."
        ),
    )


class Plan_metadata(BaseModel):
    """Pipeline-level metadata describing the task as a whole."""

    complexity: Literal["low", "medium", "high"] = Field(
        description=(
            "Estimated complexity of the task:\n"
            "- 'low': 1-2 steps, no joins, single table/file, simple aggregation.\n"
            "- 'medium': 3-5 steps, may include joins, multiple tables, basic transformations.\n"
            "- 'high': 6+ steps, multi-stage pipeline, complex joins, business logic, "
            "or requires domain-specific calculations."
        ),
    )

    required_output: str = Field(
        description=(
            "Final output filename WITH extension, including the 'data/' folder prefix. "
            "Examples: 'data/report.xlsx', 'data/summary.csv', 'data/result.json'. "
            "Default extension is .xlsx unless the user explicitly requested CSV or JSON."
        ),
    )


class Plan_Schema(BaseModel):
    """The complete execution plan returned by the planner.
    Must contain metadata (one object) and at least one step in the steps list.
    """

    metadata: Plan_metadata = Field(
        description="Pipeline-level metadata describing task complexity and final output file.",
    )

    steps: list[Plan_steps] = Field(
        description=(
            "Ordered list of execution steps. Must contain AT LEAST one step. "
            "Steps must be in execution order (step_id 1 first). "
            "Only the LAST step may have action='save_to_file'."
        ),
        min_length=1,
    )


# ===================== EXTRACTING =====================

class Extra_Schema(BaseModel):
    """Result of analyzing the user's task to decide how to route it.
    Returned by the extractor node before planning begins.
    """

    type: Literal["sql", "file"] = Field(
        description=(
            "How to handle the user's task:\n"
            "- 'sql': the task requires data from the SQL database. Use when the user "
            "asks about clients, transactions, accounts, sales, balances — anything that "
            "lives in the warehouse.\n"
            "- 'file': the task requires working with a local file (CSV/XLSX) the user "
            "has uploaded or named. Use when the user explicitly references a filename "
            "or the task is clearly about a local dataset.\n"
            "Choose exactly one. Do not default to 'sql' if both could apply."
        ),
    )

    confidence: float = Field(
        description=(
            "How confident you are in your routing decision, on a 0.0 to 1.0 scale. "
            "Use 0.9+ when the task clearly maps to one type. "
            "Use 0.6-0.8 when ambiguous but leaning toward one option. "
            "Use below 0.6 ONLY if you genuinely cannot tell — note that the system "
            "rejects plans with confidence < 0.6, so reserve low scores for real ambiguity."
        ),
        ge=0.0,
        le=1.0,
    )

    original_query: str = Field(
        description=(
            "The user's original request, copied VERBATIM with no edits, no paraphrasing, "
            "no translation. This is what the planner will see as the user-facing request."
        ),
    )

    translated_query: str = Field(
        description=(
            "A cleaned, unambiguous rewrite of the user's query optimized for downstream "
            "agents. Keep the same meaning but: fix typos, expand abbreviations, resolve "
            "ambiguous pronouns ('it' → 'the sales table'), and make implicit context "
            "explicit. Same language as the original. "
            "Example: 'how many clients from msk?' → 'Count distinct clients located "
            "in Moscow region from the clients table.'"
        ),
    )

    file_name: str = Field(
        description=(
            "Name of the file (with extension) the task refers to. "
            "Set this only when type='file' AND the user named a specific file. "
            "If type='file' but no filename is in the task, leave an empty string. "
            "If type='sql', leave an empty string."
        ),
    )


# ===================== TABLES SELECTING =====================

class Selected_Tables(BaseModel):
    """One table chosen by the selector to satisfy the user's query."""

    table_name: str = Field(
        description=(
            "Exact name of the selected table as it exists in the database. "
            "Match the name character-for-character from the candidates you were given. "
            "Do not invent or rename tables."
        ),
    )

    reason: str = Field(
        description=(
            "Concise (1-2 sentences) explanation of WHY this table is needed for the "
            "user's query. Mention the relevant columns or join keys. "
            "Example: 'Contains transaction amounts and client_id needed for the join "
            "with the clients table.'"
        ),
    )


class SQL_tables_Schema(BaseModel):
    """Result of the selector choosing which tables to use from a candidate set.
    Returned by the selector node after table reranking.
    """

    selected_tables: list[Selected_Tables] = Field(
        description=(
            "List of tables required to satisfy the query. Include EVERY table that "
            "must be joined or read — don't omit intermediate tables even if the user "
            "didn't name them. Order does not matter. "
            "If NONE of the candidate tables fit the query (genuine mismatch), return "
            "exactly one entry with table_name='failure' and reason explaining "
            "what was missing — this triggers the system to widen the candidate pool."
        ),
        min_length=1,
    )