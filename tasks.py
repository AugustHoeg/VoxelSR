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
def trainid(ctx, model, dataset, experiment_id, dataset_path=None):
    ctx.run(
        f"python -u train.py "
        f"-cn {model} "
        f"dataset_opt={dataset} "
        f"experiment_id={model}_{dataset}_{experiment_id} "
        f"dataset_opt.dataset_path={dataset_path if dataset_path is not None else '../3D_datasets/datasets/'}"
    )

@task
def finetune(ctx, model, dataset, experiment_id, pretrained_experiment_id):
    """Run the training script."""
    ctx.run(f"python -u train.py -cn {model} dataset_opt={dataset} experiment_id={experiment_id} train_mode='finetune' path.pretrained_experiment_id={pretrained_experiment_id}")


@task
def finetuneid(ctx, model, dataset, experiment_id, pretrained_experiment_id):
    """Run the training script."""
    ctx.run(f"python -u train.py -cn {model} dataset_opt={dataset} experiment_id={model}_{dataset}_{experiment_id} train_mode='finetune' path.pretrained_experiment_id={model}_{dataset}_{pretrained_experiment_id}")

@task
def testzarr(ctx, experiment_id):
    """Run the testing script."""
    ctx.run(f"python -u inference_zarr.py experiment_id={experiment_id}")

def testzarrid(ctx, model, dataset, experiment_id, dataset_path=None):
    ctx.run(
        f"python -u inference_zarr.py "
        f"experiment_id={model}_{dataset}_{experiment_id} "
        f"dataset_opt.dataset_path={dataset_path if dataset_path is not None else '../3D_datasets/datasets/'}"
    )

@task
def testzarrcross(ctx, experiment_id, datasets, mode):
    """Run the testing script."""
    ctx.run(f"python -u inference_zarr_cross.py experiment_id={experiment_id} dataset_opt.datasets={datasets} dataset_opt.synthetic={True if mode == 'synthetic' else False}")

@task
def LAM(ctx, experiment_id, datasets, cube_no, window_size, h, w, d):
    """Run the testing script."""
    with ctx.cd("LAM_3d/"):
        ctx.run(f"python -u LAM_3d_anymodel_padding.py experiment_id={experiment_id} dataset_opt.datasets={datasets} +LAM_opt.cube_no={cube_no} +LAM_opt.window_size={window_size} +LAM_opt.h={h} +LAM_opt.w={w} +LAM_opt.d={d} +LAM_opt.use_new_cube_dir={True}")


@task
def nsysproftrain(ctx, model, dataset, experiment_id):
    """Run the testing script."""
    ctx.run(f"nsys profile -o my_profile_report python train.py experiment_id={model}_{dataset}_{experiment_id}")

@task
def generatecubes(ctx, dataset, mode):
    """Run generate cubes script."""
    ctx.run(f"python -u LAM_3d/generate_cubes.py -cn 'generate_cubes' dataset_opt.datasets={dataset} dataset_opt.synthetic={True if mode == 'synthetic' else False}")
