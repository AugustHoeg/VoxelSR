from invoke import task

@task
def template(ctx):
    """Create a new project from the template."""
    ctx.run("cookiecutter -f --no-input --verbose .")

@task
def requirements(ctx):
    """Install project requirements."""
    ctx.run("python -m pip install --upgrade pip")
    ctx.run("pip install -r requirements.txt")

@task
def train(ctx, model, dataset):
    """Run the training script."""
    ctx.run(f"python train.py -cn {model} dataset_opt={dataset}")

@task
def test(ctx, experiment_id):
    """Run the testing script."""
    ctx.run(f"python test.py experiment_id={experiment_id}")

