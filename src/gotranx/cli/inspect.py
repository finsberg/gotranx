import typer

app = typer.Typer()


@app.command()
def main():
    typer.echo("Hello from inspect")
