# Stage 1: Build the virtual environment
FROM python:3.12-slim as builder

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Install uv, the Python package installer
RUN pip install uv

# Set up a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies into the virtual environment from the lock file
# Using uv sync with the lock file is the recommended and fastest way
RUN uv pip sync uv.lock

# Stage 2: Create the final application image
FROM python:3.12-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add the virtual environment to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
# The project uses a src layout
COPY ./src ./src
COPY ./scripts ./scripts
COPY ./data ./data
COPY ./assets ./assets
COPY ./pyproject.toml ./

# Expose ports for the API and Streamlit UI
EXPOSE 8000
EXPOSE 8501

# The CMD will be set in the docker-compose.yml to allow for flexibility
# For example, to run the API:
# CMD ["uvicorn", "src.autovisionai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Or to run the UI:
# CMD ["streamlit", "run", "src/autovisionai/ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
