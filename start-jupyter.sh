#!/bin/bash

# Automatically trust all notebooks under /work/notebooks
if [ -d /work/notebooks ]; then
  find /work/notebooks -type f -name "*.ipynb" -exec jupyter trust {} \;
fi

# Start the usual JupyterLab server
exec start-notebook.sh \
  --NotebookApp.token='' \
  --NotebookApp.allow_origin='*' \
  --NotebookApp.notebook_dir=/work
