FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY code ./code
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["ovacast-train"]
