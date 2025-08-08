from invoke import task
from datetime import datetime
@task
def git(ctx, message=None):
    """Run the testing script."""
    ctx.run(f"git add .")
    if message is None:
        message = "update"
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push origin main")

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
    ctx.run(f"python -u train.py -cn {model} dataset_opt={dataset}")

@task
def trainid(ctx, model, dataset, experiment_id):
    """Run the training script."""
    ctx.run(f"python -u train.py -cn {model} dataset_opt={dataset} experiment_id={model}_{dataset}_{experiment_id}")

@task
def finetuneid(ctx, model, dataset, experiment_id, pretrained_experiment_id):
    """Run the training script."""
    ctx.run(f"python -u train.py -cn {model} dataset_opt={dataset} experiment_id={model}_{dataset}_{experiment_id} train_mode='finetune' path.pretrained_experiment_id={model}_{dataset}_{pretrained_experiment_id}")

@task
def test(ctx, experiment_id):
    """Run the testing script."""
    ctx.run(f"python -u test.py experiment_id={experiment_id}")

@task
def testid(ctx, model, dataset, experiment_id):
    """Run the testing script."""
    ctx.run(f"python -u test.py experiment_id={model}_{dataset}_{experiment_id}")

@task
def runtest(ctx, experiment_id):
    """Run the testing script."""
    ctx.run(f"python -u run_test.py experiment_id={experiment_id}")

@task
def runtestid(ctx, model, dataset, experiment_id):
    """Run the testing script."""
    ctx.run(f"python -u run_test.py experiment_id={model}_{dataset}_{experiment_id}")

@task
def inferencezarr(ctx, model, dataset, experiment_id):
    """Run the testing script."""
    ctx.run(f"python -u inference_zarr.py experiment_id={model}_{dataset}_{experiment_id}")

@task
def nsysproftrain(ctx, model, dataset, experiment_id):
    """Run the testing script."""
    ctx.run(f"nsys profile -o my_profile_report python train.py experiment_id={model}_{dataset}_{experiment_id}")

