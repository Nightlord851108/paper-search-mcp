# Multi-platform Docker image for Paper Search MCP
FROM python:3.10-alpine

# Install system dependencies
RUN apk add --no-cache build-base libffi-dev openssl-dev

WORKDIR /app

# Copy the entire repository
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

# Expose port for HTTP mode
EXPOSE 3000

# Set environment variables
ENV MCP_TRANSPORT=http
ENV PORT=3000

# Command to run the MCP server
CMD ["python", "-m", "paper_search_mcp.server"]
