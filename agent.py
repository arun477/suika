import typer
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import os
import torch
import torch._dynamo
import json
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
import warnings
import logging
import contextlib
from PIL import Image


warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)
torch._dynamo.config.suppress_errors = True
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("triton").setLevel(logging.CRITICAL)

MODEL_NAME = "answerdotai/ModernBERT-base"
META_FILE = 'meta.json'
console = Console()

@contextlib.contextmanager
def suppress_outputs():
    """Context manager to suppress all outputs, warnings, and errors temporarily"""
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

def load_model(model_name):
    with suppress_outputs():
        model = SentenceTransformer(model_name)
    return model


def load_meta(meta_file):
    with open(meta_file, 'r') as f:
        return json.loads(f.read())

def embed(items, model):
    return model.encode(items)
 
def load_meta_emb(meta, model):
    return embed([doc['description'] for doc in meta], model)

def get_meta_match(meta_emb, q_emb, model):
    match = torch.topk(util.pytorch_cos_sim(q_emb, meta_emb), k=1)
    match_idx, score = match.indices[0][0].item(), match.values[0][0].item()
    return match_idx, score
  
 
def ask(question, model, meta, meta_emb):
    question_emb = embed(question, model)
    match_idx, score = get_meta_match(meta_emb, question_emb, model)
    return meta[match_idx], score

def format_response(match: dict, score: float) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row(
        "[bold blue]Command:[/bold blue]",
        match.get('name', 'N/A')
    )
    if 'examples' in match and match['examples']:
        examples_md = "\n".join(f"```bash\n{example}\n```" for example in match['examples'])
        table.add_row(
            "[bold blue]Examples:[/bold blue]",
            ""
        )
        table.add_row(
            "",
            Markdown(examples_md)
        )
    return Panel(
        table,
        title=f"🦦 [bold]Match found (confidence: {score:.2f})[/bold]",
        border_style="green"
    )
 

def main():
    console.print(Panel.fit("🦦 Suika Loading ...", title="Initializing"))
    model = load_model(MODEL_NAME)
    meta = load_meta(META_FILE)
    meta_emb = load_meta_emb(meta, model)
    
    console.print(Panel.fit("✨ Ask me any linux commands (type 'exit' to quit)", title="🦦 Suika Ready"))
    while True:
        question = typer.prompt("\nYour question")
        if question.lower() == 'exit':
            console.print("\n🦦 👋 Goodbye!")
            break   
        try:
            match, score = ask(question, model, meta, meta_emb)
            response_panel = format_response(match, score)
            console.print(response_panel)
            feedback = typer.confirm("\n🦦 Was this response helpful?")
            if not feedback:
                console.print("[yellow]🦦 I'm sorry the response wasn't helpful. Please try rephrasing your question.[/yellow]")      
        except Exception as e:
            console.print(f"[red]🦦 An error occurred: {str(e)}[/red]")
            console.print("🦦 Please try again with a different question.")

if __name__ == "__main__":
    typer.run(main)