# Paper Search MCP

A Model Context Protocol (MCP) server for searching and downloading academic papers from multiple sources, including arXiv, PubMed, bioRxiv, and Sci-Hub (optional). Designed for seamless integration with large language models like Claude Desktop.

![PyPI](https://img.shields.io/pypi/v/paper-search-mcp.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
[![smithery badge](https://smithery.ai/badge/@openags/paper-search-mcp)](https://smithery.ai/server/@openags/paper-search-mcp)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Install Package](#install-package)
    - [Configure Claude Desktop](#configure-claude-desktop)
  - [For Development](#for-development)
    - [Setup Environment](#setup-environment)
    - [Install Dependencies](#install-dependencies)
- [HTTP Service](#http-service)
  - [Running HTTP Server](#running-http-server)
  - [ChatGPT Integration](#chatgpt-integration)
- [Contributing](#contributing)
- [Demo](#demo)
- [License](#license)
- [TODO](#todo)

---

## Overview

`paper-search-mcp` is a Python-based MCP server that enables users to search and download academic papers from various platforms. It provides tools for searching papers (e.g., `search_arxiv`) and downloading PDFs (e.g., `download_arxiv`), making it ideal for researchers and AI-driven workflows. Built with the MCP Python SDK, it integrates seamlessly with LLM clients like Claude Desktop.

---

## Features

- **Multi-Source Support**: Search and download papers from arXiv, PubMed, bioRxiv, medRxiv, Google Scholar, IACR ePrint Archive, Semantic Scholar.
- **Standardized Output**: Papers are returned in a consistent dictionary format via the `Paper` class.
- **Asynchronous Tools**: Efficiently handles network requests using `httpx`.
- **MCP Integration**: Compatible with MCP clients for LLM context enhancement.
- **Extensible Design**: Easily add new academic platforms by extending the `academic_platforms` module.

---

## Installation

`paper-search-mcp` can be installed using `uv` or `pip`. Below are two approaches: a quick start for immediate use and a detailed setup for development.

### Installing via Smithery

To install paper-search-mcp for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@openags/paper-search-mcp):

```bash
npx -y @smithery/cli install @openags/paper-search-mcp --client claude
```

### Quick Start

For users who want to quickly run the server:

1. **Install Package**:

   ```bash
   uv add paper-search-mcp
   ```

2. **Configure Claude Desktop**:
   Add this configuration to `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):
   ```json
   {
     "mcpServers": {
       "paper_search_server": {
         "command": "uv",
         "args": [
           "run",
           "--directory",
           "/path/to/your/paper-search-mcp",
           "-m",
           "paper_search_mcp.server"
         ],
         "env": {
           "SEMANTIC_SCHOLAR_API_KEY": "" // Optional: For enhanced Semantic Scholar features
         }
       }
     }
   }
   ```
   > Note: Replace `/path/to/your/paper-search-mcp` with your actual installation path.

### For Development

For developers who want to modify the code or contribute:

1. **Setup Environment**:

   ```bash
   # Install uv if not installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone repository
   git clone https://github.com/openags/paper-search-mcp.git
   cd paper-search-mcp

   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies**:

   ```bash
   # Install project in editable mode
   uv add -e .

   # Add development dependencies (optional)
   uv add pytest flake8
   ```

---

## HTTP Service

`paper-search-mcp` supports both stdio and HTTP transport modes, making it compatible with ChatGPT and other HTTP-based clients.

### Running HTTP Server

To run the server in HTTP mode:

1. **Set Environment Variable**:
   ```bash
   export MCP_TRANSPORT=http
   export PORT=3000  # Optional: defaults to 3000
   ```

2. **Start the Server**:
   ```bash
   # Using uv
   uv run -m paper_search_mcp.server

   # Or with python directly
   python -m paper_search_mcp.server
   ```

3. **Test the Server**:
   ```bash
   # Health check
   curl http://localhost:3000

   # Ready check
   curl http://localhost:3000/ready
   ```

### ChatGPT Integration

To integrate with ChatGPT:

1. **Deploy to a Server**: Deploy the HTTP server to a cloud service (like Railway, Heroku, or AWS).

2. **Create GPT Action**: In ChatGPT, create a custom GPT with the following action schema:

   ```yaml
   openapi: 3.0.0
   info:
     title: Paper Search MCP
     version: 1.0.0
   servers:
     - url: https://your-deployed-server.com
   paths:
     /:
       post:
         summary: Execute MCP tool
         requestBody:
           content:
             application/json:
               schema:
                 type: object
                 properties:
                   jsonrpc:
                     type: string
                     enum: ["2.0"]
                   method:
                     type: string
                     enum: ["tools/call"]
                   params:
                     type: object
                     properties:
                       name:
                         type: string
                         enum: ["search_arxiv", "search_pubmed", "search_biorxiv", "search_medrxiv", "search_google_scholar", "search_iacr", "search_semantic", "search_crossref", "download_arxiv", "read_arxiv_paper"]
                       arguments:
                         type: object
                   id:
                     type: string
                 required: ["jsonrpc", "method", "params", "id"]
         responses:
           '200':
             description: Successful response
   ```

3. **Example Usage in ChatGPT**:
   - "Search for machine learning papers on arXiv"
   - "Find recent papers about neural networks from PubMed"
   - "Download and read the arXiv paper 2106.15928"

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**:
   Click "Fork" on GitHub.

2. **Clone and Set Up**:

   ```bash
   git clone https://github.com/yourusername/paper-search-mcp.git
   cd paper-search-mcp
   pip install -e ".[dev]"  # Install dev dependencies (if added to pyproject.toml)
   ```

3. **Make Changes**:

   - Add new platforms in `academic_platforms/`.
   - Update tests in `tests/`.

4. **Submit a Pull Request**:
   Push changes and create a PR on GitHub.

---

## Demo

<img src="docs\images\demo.png" alt="Demo" width="800">

## TODO

### Planned Academic Platforms

- [√] arXiv
- [√] PubMed
- [√] bioRxiv
- [√] medRxiv
- [√] Google Scholar
- [√] IACR ePrint Archive
- [√] Semantic Scholar
- [ ] PubMed Central (PMC)
- [ ] Science Direct
- [ ] Springer Link
- [ ] IEEE Xplore
- [ ] ACM Digital Library
- [ ] Web of Science
- [ ] Scopus
- [ ] JSTOR
- [ ] ResearchGate
- [ ] CORE
- [ ] Microsoft Academic

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Happy researching with `paper-search-mcp`! If you encounter issues, open a GitHub issue.
