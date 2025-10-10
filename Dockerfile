FROM jupyter/base-notebook:python-3.11

WORKDIR /work
COPY requirements.txt .

RUN mamba install -y --quiet -c conda-forge --file requirements.txt || \
    mamba install -y -c conda-forge numpy pandas matplotlib scikit-learn seaborn tqdm jupyterlab

# Ensure correct perms and LF endings for the startup script
USER root
COPY --chmod=0755 start-jupyter.sh /usr/local/bin/start-jupyter
# Fallback for older Docker engines that ignore --chmod
RUN chmod 0755 /usr/local/bin/start-jupyter && \
    sed -i 's/\r$//' /usr/local/bin/start-jupyter
USER ${NB_UID}

ENV PYTHONPATH=/work
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/start-jupyter"]
