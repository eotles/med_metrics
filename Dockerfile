FROM jupyter/base-notebook:python-3.11

WORKDIR /work
COPY requirements.txt .

# use conda to install packages into the running jupyter env
RUN mamba install -y --quiet -c conda-forge \
    --file requirements.txt || \
    mamba install -y -c conda-forge numpy pandas matplotlib scikit-learn seaborn tqdm jupyterlab

# add startup script to trust notebooks
USER root
COPY start-jupyter.sh /usr/local/bin/start-jupyter
RUN chmod +x /usr/local/bin/start-jupyter
USER ${NB_UID}

ENV PYTHONPATH=/work
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/start-jupyter"]
