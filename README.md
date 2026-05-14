# RLMproved
A collection of everything needed to run the solutions proposed in RLMproved.

## Project Structure

The project consists of two folders:
- rlm-api: contains an api to try parsed_responses and parsed_completions.
- rlm-mcp-server: contains the MCP server and the AI application API.

## Prerequisites

Python 3.12 is required.

Before getting started, make sure you have Poetry installed on your system.

Poetry is used to manage project dependencies and virtual environments. You can check whether Poetry is already installed by running:

```bash
poetry --version
```

If Poetry is not installed, follow the official installation instructions, then try the previous command again:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Installation

### MCP

To install the local environment for the MCP application you need to run the following commands.

First let's install and run the mcp server.
```bash
cd rlm-mcp-server
poetry install
poetry run python ./src/rlm_mcp/rlm_mcp_server/mcp_main.py
``` 

And in a separate shell/process:
```
poetry run uvicorn rlm_mcp.rlm_mcp_client.api_main:app --reload --port 6969
```

You can then access the fastAPI at this link:

```bash
http://127.0.0.1:6969/docs
```


### Formatted Completions and Responses

First make sure you are in:

```bash
cd rlm-api
```

Then you want to:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
poetry env use $(which python)
```

Make sure that you are using python3.12 by 

```bash
poetry run python --version
```

You should see Python 3.12.13 for example.

Then:
```bash
poetry install
poetry run fastapi run
```


You can then access the fastAPI at this link:
```bash
http://127.0.0.1:8000/docs
```
