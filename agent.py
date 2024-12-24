import typer
from sentence_transformers import SentenceTransformer, util
import os
import torch
import torch._dynamo
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
import warnings
import logging
import contextlib
import db as _db

from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.console import Group
from rich.syntax import Syntax
from rich.style import Style
from rich.text import Text
from rich.prompt import Prompt


# supress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
torch._dynamo.config.suppress_errors = True
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("triton").setLevel(logging.CRITICAL)


@contextlib.contextmanager
def suppress_outputs():
    """Context manager to suppress all outputs, warnings, and errors temporarily"""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


# model infos
MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_NAME = "all-mpnet-base-v2"
SUIK_LOGO = "🦦"


# agent config
META_FILE = "meta.json"
console = Console()
LABEL_COLORS = Style(color="#B7D46F", bold=True)
LABEL_COLORS = Style(color="#808080", bold=True)


def load_model(model_name):
    with suppress_outputs():
        model = SentenceTransformer(model_name)
    return model


def load_meta(meta_file):
    return _db.fetch_all_documents()
    with open(meta_file, "r") as f:
        return json.loads(f.read())


def embed(items, model):
    return model.encode(items)


def load_meta_emb(meta, model):
    return embed([doc["description"] for doc in meta], model)


def get_meta_match(meta_emb, q_emb, model):
    match = torch.topk(util.pytorch_cos_sim(q_emb, meta_emb), k=1)
    match_idx, score = match.indices[0][0].item(), match.values[0][0].item()
    return match_idx, score


def ask(question, model, meta, meta_emb):
    question_emb = embed(question, model)
    match_idx, score = get_meta_match(meta_emb, question_emb, model)
    return meta[match_idx], score


def format_response(match: dict, score: float) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1), collapse_padding=True)

    cmd_text = Text.assemble(("Command: ", LABEL_COLORS), (match.get("name", "N/A"), Style(color="white")))
    table.add_row(cmd_text)

    if description := match.get("description"):
        desc_text = Text.assemble(("Description: ", LABEL_COLORS), (description, Style(color="white")))
        table.add_row(desc_text)

    if examples := match.get("examples"):
        table.add_row(Text("Examples:", style=LABEL_COLORS))
        for idx, example in enumerate(examples, start=1):
            example_label = Text.assemble((f"  Example command: ", Style(dim=True)))
            table.add_row(example_label)
            code = Syntax(
                "    " + example, "bash", theme="material", line_numbers=False, word_wrap=True, padding=(0, 1), background_color="default"
            )
            table.add_row(code)
    footer = Text.assemble(
        ("Model: ", LABEL_COLORS),
        (MODEL_NAME, Style(color="white", italic=True)),
        (" | Match Score: ", LABEL_COLORS),
        (f"{score:.2f}", Style(color="white")),
    )

    return Panel(
        table,
        title=f"{SUIK_LOGO} Match",
        subtitle=footer,
        border_style="white",
        padding=(0, 1),
    )


def main():
    console.print(Panel.fit(f"{SUIK_LOGO} Suika Loading ...", title="Initializing"))
    model = load_model(MODEL_NAME)
    meta = load_meta(META_FILE)
    meta_emb = load_meta_emb(meta, model)
    console.print(Panel.fit("✨ Ask me any linux commands (type 'exit' to quit)", title=f"{SUIK_LOGO} Suika Ready"))
    while True:
        question = Prompt.ask(f"[#B7D46F]Your question {SUIK_LOGO} ")
        if question.lower() == "exit":
            console.print(f"{SUIK_LOGO} 👋 Goodbye!")
            break
        try:
            match, score = ask(question, model, meta, meta_emb)
            response_panel = format_response(match, score)
            console.print(response_panel)
        except Exception as e:
            console.print(f"[red]{SUIK_LOGO} An error occurred: {str(e)}[/red]")
            console.print(f"{SUIK_LOGO} Please try again with a different question.")


if __name__ == "__main__":
    typer.run(main)
