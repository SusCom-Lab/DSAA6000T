# DSAA6000T Course Space

This repository is the shared workspace for DSAA6000T course materials. It is intended for lecture notes, runnable notebooks, examples, and supporting resources that students can use and improve together.

## Course contents

| Week | Topic | Materials |
| --- | --- | --- |
| 1 | Profiling the components of a Llama 8B model | [Week 1 guide](week1/README.md) · [Jupyter notebook](week1/llama8b_component_profiling.ipynb) |

## Getting started

Create the course environment from the repository root:

```bash
conda env create -f environment.yml
conda activate dsaa6000t-week1
python -m ipykernel install --user \
  --name dsaa6000t-week1 \
  --display-name "Python (DSAA6000T Week 1)"
jupyter lab
```

Students provide their own local model directory when the Week 1 notebook asks for it. The repository does not contain model weights or machine-specific paths.

## Sharing guidelines

- Put each week's materials in its corresponding `weekN/` directory.
- Keep notebooks runnable from a clean environment.
- Do not commit model weights, generated traces, profiling results, credentials, or personal filesystem paths.
- Document any additional dependency required by an exercise.

