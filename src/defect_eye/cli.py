import typer

app = typer.Typer(
    name="defect-eye",
    help="AI-driven Software Defect Predictor CLI tool.",
)

@app.command()
def version():
    """Display installed version."""
    typer.echo("defect-eye version 0.1.0")

if __name__ == "__main__":
    app()