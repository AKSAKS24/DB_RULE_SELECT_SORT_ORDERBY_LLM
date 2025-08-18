# app_orderby_sort_strict.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import os, re, json

# ---- LLM Setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="ABAP Scanner - ORDER BY/SORT Strict")

# ===== Regex for detection =====
SQL_SELECT_BLOCK_RE = re.compile(
    r"\bSELECT\b(?P<select>.+?)\bFROM\b\s+(?P<table>\w+)(?P<rest>.*?\.)",
    re.IGNORECASE | re.DOTALL,
)
FOR_ALL_ENTRIES_RE = re.compile(r"\bFOR\s+ALL\s+ENTRIES\b", re.IGNORECASE)
FIELDS_RE = re.compile(r"\b(\w+)\b", re.IGNORECASE)
ORDERBY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
INTO_TABLE_RE = re.compile(
    r"\bINTO\s+TABLE\s+(?:@DATA\((\w+)\)|(\w+))", re.IGNORECASE
)

def make_sort_re(table_name: str):
    return re.compile(rf"\bSORT\s+{re.escape(table_name)}\b[\s\S]*?\bBY\b", re.IGNORECASE)


# ===== Strict Models =====
class SelectItem(BaseModel):
    table: Optional[str]
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def filter_none(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]


class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = None
    code: Optional[str] = ""
    selects: List[SelectItem] = Field(default_factory=list)


# ===== Detection logic =====
def parse_and_build_selectitems(code: str) -> List[SelectItem]:
    results: List[SelectItem] = []

    for stmt in SQL_SELECT_BLOCK_RE.finditer(code):
        stmt_text = stmt.group(0)
        span = stmt.span()
        has_fae = FOR_ALL_ENTRIES_RE.search(stmt_text) is not None

        # Extract selected fields
        select_fields_raw = stmt.group("select")
        select_fields_raw = re.sub(
            r"\bINTO\b.+", "", select_fields_raw,
            flags=re.IGNORECASE | re.DOTALL
        )
        select_fields_raw = re.sub(r"\s+", " ", select_fields_raw).strip()

        # Extract fields list
        fields = []
        for tok in FIELDS_RE.findall(select_fields_raw):
            tok_up = tok.upper()
            if tok_up not in ["DISTINCT"]:
                fields.append(tok_up)
        fields = list(dict.fromkeys(fields))

        # Extract INTO TABLE target
        target_table = None
        m = INTO_TABLE_RE.search(stmt_text)
        if m:
            target_table = m.group(1) or m.group(2)

        suggestion = None

        if not has_fae:  # Normal SELECT
            if not ORDERBY_RE.search(stmt_text.replace("\n", " ")):
                suggestion = (
                    f"Add ORDER BY {', '.join(fields)} inside SELECT (all fields in select list)."
                    if fields else "Add ORDER BY with all select fields."
                )
        else:  # FOR ALL ENTRIES
            after_stmt_text = code[span[1]:]

            # Clean comments
            cleaned_lines = []
            for line in after_stmt_text.splitlines():
                if line.strip().startswith("*"):
                    continue
                cleaned_lines.append(line)
            after_stmt_text = "\n".join(cleaned_lines)

            found_sort = False
            if target_table:
                if make_sort_re(target_table).search(after_stmt_text):
                    found_sort = True

            if not found_sort:
                suggestion = (
                    f"Add SORT by {', '.join(fields)} after this SELECT into {target_table} (all fields)."
                    if fields else f"Add SORT by all select fields after this SELECT into {target_table}."
                )

        if suggestion:
            results.append(
                SelectItem(
                    table=target_table,
                    target_type="SQL_SELECT",
                    target_name="FOR_ALL_ENTRIES" if has_fae else "NO_FOR_ALL_ENTRIES",
                    used_fields=fields,
                    suggested_fields=[],
                    suggested_statement=suggestion
                )
            )
    return results


# ===== Summarizer =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    findings = []
    for s in unit.selects:
        findings.append({
            "table": s.table,
            "reason": s.suggested_statement,
            "fields": s.used_fields
        })
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_occurrences": len(unit.selects),
            "orderby_sort_flags": findings
        }
    }


# ===== LLM Prompt =====
SYSTEM_MSG = "You are a precise ABAP reviewer for ORDER BY / SORT rules. Output strict JSON only."
USER_TEMPLATE = """
You are assessing ABAP code usage in light of ORDER BY / SORT rules.

For each SELECT:
- Normal SELECT must have ORDER BY.
- SELECT ... FOR ALL ENTRIES must be followed by SORT of the same table.

We provide metadata and findings (under "selects").
Tasks:
1) Produce a concise assessment of violations.
2) Produce an actionable LLM remediation prompt (e.g. add ORDER BY/SORT).

Return ONLY strict JSON:
{{
  "assessment": "<concise summary>",
  "llm_prompt": "<remediation action prompt>"
}}

Unit metadata:
- Program: {pgm}
- Include: {inc}
- Unit type: {utype}
- Unit name: {uname}

Summary:
{plan_json}

Findings:
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser


def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan_json = json.dumps(summarize_selects(unit), indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], indent=2)
    try:
        return chain.invoke({
            "pgm": unit.pgm_name,
            "inc": unit.inc_name,
            "utype": unit.type,
            "uname": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}")


# ===== API Endpoint =====
@app.post("/assess-orderby-sort-strict")
def assess_orderby_sort(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        if u.code:
            u.selects = parse_and_build_selectitems(u.code)

        if not u.selects:
            continue

        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)
        obj.pop("code", None)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)
    return out


@app.get("/health")
def health():
    return {"ok": True, "rule": "ORDERBY_SORT", "model": OPENAI_MODEL}