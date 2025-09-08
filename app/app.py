# app_2198647_suggestion_only.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json

# --- LLM ENV ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="ABAP Suggestion API - Note 2198647 (Suggestion Only)")

# --- MODELS ---

class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    findings: Optional[List[Finding]] = Field(default_factory=list)

# --- LLM PROMPT ---

SYSTEM_MSG = """
You are an ABAP upgrade advisor. Output ONLY valid JSON as response.
- For every provided .findings[] with a non-empty "suggestion":
  - Write a bullet point using ONLY the "suggestion" field as the corrective action.
  - If "snippet" is non-empty, insert it (as ABAP code/text) before/after the suggestion where it fits.
  - Do not reference or require code outside "snippet".
  - Omit findings without a suggestion.
- Cover ALL findings with suggestions.
Return JSON (and nothing else) with:
{{
  "assessment": "<summary of  issues>",
  "llm_prompt": "<bulleted list with all actions as described above>"
}}
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Class implementation: {class_implementation}
Start line: {start_line}
End line: {end_line}

findings (JSON, each with all fields, filtered for 2198647-related issues):
{findings_json}

Instructions:
1. Write a summary ("assessment").
2. For every finding containing a non-empty suggestion, add a bullet in "llm_prompt":
    - Use the "suggestion" field as the action text.
    - If "snippet" (code) is present and non-empty, include it alongside the action (in-line).
    - Skip any findings without a suggestion.
Return valid JSON:
{{
  "assessment": "<concise 2198647 impact paragraph>",
  "llm_prompt": "<bullet list of actionable suggestions>"
}}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    # Only findings with a non-empty suggestion
    relevant = [f for f in (unit.findings or []) if f.suggestion and f.suggestion.strip()]
    if not relevant:
        return None
    findings_json = json.dumps([f.model_dump() for f in relevant], ensure_ascii=False, indent=2)
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "class_implementation": unit.class_implementation or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-orderby-sort-strict")
async def assess_2198647_suggestions(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        llm_out = llm_assess_and_prompt(u)
        if not llm_out:
            continue
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj = {
            "pgm_name": u.pgm_name,
            "inc_name": u.inc_name,
            "type": u.type,
            "name": u.name,
            "class_implementation": u.class_implementation,
            "start_line": u.start_line,
            "end_line": u.end_line,
            "assessment": llm_out.get("assessment", ""),
            "llm_prompt": prompt_out
        }
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "note": "2198647", "model": OPENAI_MODEL}