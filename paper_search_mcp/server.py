# paper_search_mcp/server.py
from typing import List, Dict, Optional
import httpx
import asyncio
import logging
import os
import json
import hashlib
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from mcp.server.fastmcp import FastMCP
from .academic_platforms.arxiv import ArxivSearcher
from .academic_platforms.pubmed import PubMedSearcher
from .academic_platforms.biorxiv import BioRxivSearcher
from .academic_platforms.medrxiv import MedRxivSearcher
from .academic_platforms.google_scholar import GoogleScholarSearcher
from .academic_platforms.iacr import IACRSearcher
from .academic_platforms.semantic import SemanticSearcher
from .academic_platforms.crossref import CrossRefSearcher

# from .academic_platforms.hub import SciHubSearcher
from .paper import Paper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MCP server
mcp = FastMCP("paper_search_server")

# Initialize Flask app for HTTP transport
app = Flask(__name__)
CORS(app)

# In-memory storage for search results (simulating OpenAI Vector Storage)
search_cache: Dict[str, Dict[str, any]] = {}

# Instances of searchers
arxiv_searcher = ArxivSearcher()
pubmed_searcher = PubMedSearcher()
biorxiv_searcher = BioRxivSearcher()
medrxiv_searcher = MedRxivSearcher()
google_scholar_searcher = GoogleScholarSearcher()
iacr_searcher = IACRSearcher()
semantic_searcher = SemanticSearcher()
crossref_searcher = CrossRefSearcher()
# scihub_searcher = SciHubSearcher()

def generate_object_id(content: str) -> str:
    """Generate a unique ID for content using hash"""
    return hashlib.md5(content.encode()).hexdigest()[:12]

def store_search_results(results: List[Dict], query: str, source: str = "multi") -> List[str]:
    """Store search results and return list of object IDs"""
    object_ids = []
    for result in results:
        content_key = f"{result.get('title', '')}-{result.get('authors', '')}-{source}"
        obj_id = generate_object_id(content_key)

        # Store with enhanced metadata for better search
        search_cache[obj_id] = {
            "id": obj_id,
            "title": result.get("title", ""),
            "authors": result.get("authors", ""),
            "abstract": result.get("abstract", ""),
            "url": result.get("url", ""),
            "doi": result.get("doi", ""),
            "publication_date": result.get("publication_date", ""),
            "source": source,
            "query_used": query,
            "stored_at": datetime.now().isoformat(),
            "type": "academic_paper",
            "relevance_score": len(results) - len(object_ids)  # Simple scoring
        }
        object_ids.append(obj_id)

    return object_ids

# Standard MCP search tools (ChatGPT compatible)
@mcp.tool()
async def search(query: str, max_results: int = 10) -> Dict[str, any]:
    """
    Primary search tool that returns object IDs for academic papers (OpenAI MCP standard).
    This tool searches across multiple academic platforms and returns unified results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Dictionary containing object IDs and search metadata
    """
    logging.info(f"MCP Search tool called with query: {query}, max_results: {max_results}")

    try:
        all_results = []

        # Search across multiple platforms
        try:
            arxiv_results = await async_search(arxiv_searcher, query, min(max_results, 3))
            for result in arxiv_results:
                result['source'] = 'arxiv'
            all_results.extend(arxiv_results)
        except Exception as e:
            logging.warning(f"ArXiv search failed: {e}")

        try:
            semantic_results = await async_search(semantic_searcher, query, min(max_results, 3))
            for result in semantic_results:
                result['source'] = 'semantic_scholar'
            all_results.extend(semantic_results)
        except Exception as e:
            logging.warning(f"Semantic Scholar search failed: {e}")

        try:
            crossref_results = await async_search(crossref_searcher, query, min(max_results, 4))
            for result in crossref_results:
                result['source'] = 'crossref'
            all_results.extend(crossref_results)
        except Exception as e:
            logging.warning(f"CrossRef search failed: {e}")

        # Limit results and remove duplicates
        all_results = all_results[:max_results]

        if not all_results:
            return {
                "object_ids": [],
                "query": query,
                "total_results": 0,
                "error": "No results found across all platforms"
            }

        # Store results and get object IDs
        object_ids = store_search_results(all_results, query, "multi_platform")

        return {
            "object_ids": object_ids,
            "query": query,
            "total_results": len(object_ids),
            "search_timestamp": datetime.now().isoformat(),
            "sources_searched": ["arxiv", "semantic_scholar", "crossref"]
        }

    except Exception as e:
        logging.error(f"Error in search tool: {str(e)}")
        return {
            "object_ids": [],
            "query": query,
            "total_results": 0,
            "error": f"Search failed: {str(e)}"
        }

@mcp.tool()
async def fetch(id: str) -> Dict[str, any]:
    """
    Fetch detailed paper information using object ID (OpenAI MCP standard).

    Args:
        id: Object ID to retrieve

    Returns:
        Detailed paper information
    """
    logging.info(f"MCP Fetch tool called with id: {id}")

    if id in search_cache:
        return search_cache[id]
    else:
        return {
            "id": id,
            "error": f"Object ID {id} not found in cache"
        }

@mcp.tool()
async def search_cached(query_terms: str, limit: int = 5) -> List[Dict[str, any]]:
    """
    Search through cached results for specific terms.
    Useful for filtering already retrieved papers.

    Args:
        query_terms: Terms to search for in cached results
        limit: Maximum number of results to return

    Returns:
        List of matching cached papers
    """
    logging.info(f"Searching cached results for: {query_terms}")

    matching_results = []
    query_lower = query_terms.lower()

    for obj_id, paper in search_cache.items():
        # Search in title, abstract, and authors
        searchable_text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('authors', '')}".lower()

        if query_lower in searchable_text:
            matching_results.append(paper)

        if len(matching_results) >= limit:
            break

    # Sort by relevance score (higher is better)
    matching_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    return matching_results[:limit]


# Asynchronous helper to adapt synchronous searchers
async def async_search(searcher, query: str, max_results: int, **kwargs) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        # Assuming searchers use requests internally; we'll call synchronously for now
        if 'year' in kwargs:
            papers = searcher.search(query, year=kwargs['year'], max_results=max_results)
        else:
            papers = searcher.search(query, max_results=max_results)
        return [paper.to_dict() for paper in papers]


# Tool definitions
@mcp.tool()
async def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search academic papers from arXiv.

    Args:
        query: Search query string (e.g., 'machine learning').
        max_results: Maximum number of papers to return (default: 10).
    Returns:
        List of paper metadata in dictionary format.
    """
    papers = await async_search(arxiv_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """Search academic papers from PubMed.

    Args:
        query: Search query string (e.g., 'machine learning').
        max_results: Maximum number of papers to return (default: 10).
    Returns:
        List of paper metadata in dictionary format.
    """
    papers = await async_search(pubmed_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_biorxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search academic papers from bioRxiv.

    Args:
        query: Search query string (e.g., 'machine learning').
        max_results: Maximum number of papers to return (default: 10).
    Returns:
        List of paper metadata in dictionary format.
    """
    papers = await async_search(biorxiv_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_medrxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search academic papers from medRxiv.

    Args:
        query: Search query string (e.g., 'machine learning').
        max_results: Maximum number of papers to return (default: 10).
    Returns:
        List of paper metadata in dictionary format.
    """
    papers = await async_search(medrxiv_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_google_scholar(query: str, max_results: int = 10) -> List[Dict]:
    """Search academic papers from Google Scholar.

    Args:
        query: Search query string (e.g., 'machine learning').
        max_results: Maximum number of papers to return (default: 10).
    Returns:
        List of paper metadata in dictionary format.
    """
    papers = await async_search(google_scholar_searcher, query, max_results)
    return papers if papers else []


@mcp.tool()
async def search_iacr(
    query: str, max_results: int = 10, fetch_details: bool = True
) -> List[Dict]:
    """Search academic papers from IACR ePrint Archive.

    Args:
        query: Search query string (e.g., 'cryptography', 'secret sharing').
        max_results: Maximum number of papers to return (default: 10).
        fetch_details: Whether to fetch detailed information for each paper (default: True).
    Returns:
        List of paper metadata in dictionary format.
    """
    async with httpx.AsyncClient() as client:
        papers = iacr_searcher.search(query, max_results, fetch_details)
        return [paper.to_dict() for paper in papers] if papers else []


@mcp.tool()
async def download_arxiv(paper_id: str, save_path: str = "./downloads") -> str:
    """Download PDF of an arXiv paper.

    Args:
        paper_id: arXiv paper ID (e.g., '2106.12345').
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        Path to the downloaded PDF file.
    """
    async with httpx.AsyncClient() as client:
        return arxiv_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def download_pubmed(paper_id: str, save_path: str = "./downloads") -> str:
    """Attempt to download PDF of a PubMed paper.

    Args:
        paper_id: PubMed ID (PMID).
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        str: Message indicating that direct PDF download is not supported.
    """
    try:
        return pubmed_searcher.download_pdf(paper_id, save_path)
    except NotImplementedError as e:
        return str(e)


@mcp.tool()
async def download_biorxiv(paper_id: str, save_path: str = "./downloads") -> str:
    """Download PDF of a bioRxiv paper.

    Args:
        paper_id: bioRxiv DOI.
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        Path to the downloaded PDF file.
    """
    return biorxiv_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def download_medrxiv(paper_id: str, save_path: str = "./downloads") -> str:
    """Download PDF of a medRxiv paper.

    Args:
        paper_id: medRxiv DOI.
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        Path to the downloaded PDF file.
    """
    return medrxiv_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def download_iacr(paper_id: str, save_path: str = "./downloads") -> str:
    """Download PDF of an IACR ePrint paper.

    Args:
        paper_id: IACR paper ID (e.g., '2009/101').
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        Path to the downloaded PDF file.
    """
    return iacr_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def read_arxiv_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Read and extract text content from an arXiv paper PDF.

    Args:
        paper_id: arXiv paper ID (e.g., '2106.12345').
        save_path: Directory where the PDF is/will be saved (default: './downloads').
    Returns:
        str: The extracted text content of the paper.
    """
    try:
        return arxiv_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def read_pubmed_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Read and extract text content from a PubMed paper.

    Args:
        paper_id: PubMed ID (PMID).
        save_path: Directory where the PDF would be saved (unused).
    Returns:
        str: Message indicating that direct paper reading is not supported.
    """
    return pubmed_searcher.read_paper(paper_id, save_path)


@mcp.tool()
async def read_biorxiv_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Read and extract text content from a bioRxiv paper PDF.

    Args:
        paper_id: bioRxiv DOI.
        save_path: Directory where the PDF is/will be saved (default: './downloads').
    Returns:
        str: The extracted text content of the paper.
    """
    try:
        return biorxiv_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def read_medrxiv_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Read and extract text content from a medRxiv paper PDF.

    Args:
        paper_id: medRxiv DOI.
        save_path: Directory where the PDF is/will be saved (default: './downloads').
    Returns:
        str: The extracted text content of the paper.
    """
    try:
        return medrxiv_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def read_iacr_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Read and extract text content from an IACR ePrint paper PDF.

    Args:
        paper_id: IACR paper ID (e.g., '2009/101').
        save_path: Directory where the PDF is/will be saved (default: './downloads').
    Returns:
        str: The extracted text content of the paper.
    """
    try:
        return iacr_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def search_semantic(query: str, year: Optional[str] = None, max_results: int = 10) -> List[Dict]:
    """Search academic papers from Semantic Scholar.

    Args:
        query: Search query string (e.g., 'machine learning').
        year: Optional year filter (e.g., '2019', '2016-2020', '2010-', '-2015').
        max_results: Maximum number of papers to return (default: 10).
    Returns:
        List of paper metadata in dictionary format.
    """
    kwargs = {}
    if year is not None:
        kwargs['year'] = year
    papers = await async_search(semantic_searcher, query, max_results, **kwargs)
    return papers if papers else []


@mcp.tool()
async def download_semantic(paper_id: str, save_path: str = "./downloads") -> str:
    """Download PDF of a Semantic Scholar paper.    

    Args:
        paper_id: Semantic Scholar paper ID, Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        Path to the downloaded PDF file.
    """ 
    return semantic_searcher.download_pdf(paper_id, save_path)


@mcp.tool()
async def read_semantic_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Read and extract text content from a Semantic Scholar paper. 

    Args:
        paper_id: Semantic Scholar paper ID, Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")
        save_path: Directory where the PDF is/will be saved (default: './downloads').
    Returns:
        str: The extracted text content of the paper.
    """
    try:
        return semantic_searcher.read_paper(paper_id, save_path)
    except Exception as e:
        print(f"Error reading paper {paper_id}: {e}")
        return ""


@mcp.tool()
async def search_crossref(query: str, max_results: int = 10, **kwargs) -> List[Dict]:
    """Search academic papers from CrossRef database.
    
    CrossRef is a scholarly infrastructure organization that provides 
    persistent identifiers (DOIs) for scholarly content and metadata.
    It's one of the largest citation databases covering millions of 
    academic papers, journals, books, and other scholarly content.

    Args:
        query: Search query string (e.g., 'machine learning', 'climate change').
        max_results: Maximum number of papers to return (default: 10, max: 1000).
        **kwargs: Additional search parameters:
            - filter: CrossRef filter string (e.g., 'has-full-text:true,from-pub-date:2020')
            - sort: Sort field ('relevance', 'published', 'updated', 'deposited', etc.)
            - order: Sort order ('asc' or 'desc')
    Returns:
        List of paper metadata in dictionary format.
        
    Examples:
        # Basic search
        search_crossref("deep learning", 20)
        
        # Search with filters
        search_crossref("climate change", 10, filter="from-pub-date:2020,has-full-text:true")
        
        # Search sorted by publication date
        search_crossref("neural networks", 15, sort="published", order="desc")
    """
    papers = await async_search(crossref_searcher, query, max_results, **kwargs)
    return papers if papers else []


@mcp.tool()
async def get_crossref_paper_by_doi(doi: str) -> Dict:
    """Get a specific paper from CrossRef by its DOI.

    Args:
        doi: Digital Object Identifier (e.g., '10.1038/nature12373').
    Returns:
        Paper metadata in dictionary format, or empty dict if not found.
        
    Example:
        get_crossref_paper_by_doi("10.1038/nature12373")
    """
    async with httpx.AsyncClient() as client:
        paper = crossref_searcher.get_paper_by_doi(doi)
        return paper.to_dict() if paper else {}


@mcp.tool()
async def download_crossref(paper_id: str, save_path: str = "./downloads") -> str:
    """Attempt to download PDF of a CrossRef paper.

    Args:
        paper_id: CrossRef DOI (e.g., '10.1038/nature12373').
        save_path: Directory to save the PDF (default: './downloads').
    Returns:
        str: Message indicating that direct PDF download is not supported.
        
    Note:
        CrossRef is a citation database and doesn't provide direct PDF downloads.
        Use the DOI to access the paper through the publisher's website.
    """
    try:
        return crossref_searcher.download_pdf(paper_id, save_path)
    except NotImplementedError as e:
        return str(e)


@mcp.tool()
async def read_crossref_paper(paper_id: str, save_path: str = "./downloads") -> str:
    """Attempt to read and extract text content from a CrossRef paper.

    Args:
        paper_id: CrossRef DOI (e.g., '10.1038/nature12373').
        save_path: Directory where the PDF is/will be saved (default: './downloads').
    Returns:
        str: Message indicating that direct paper reading is not supported.
        
    Note:
        CrossRef is a citation database and doesn't provide direct paper content.
        Use the DOI to access the paper through the publisher's website.
    """
    return crossref_searcher.read_paper(paper_id, save_path)


# HTTP endpoints for MCP JSON-RPC
@app.route('/ready', methods=['GET'])
def ready():
    return jsonify({
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "service": "paper-search-mcp-server"
    })

@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
def handle_request():
    if request.method == 'GET':
        # Health check endpoint
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "paper-search-mcp-server",
            "version": "0.1.3"
        })
    elif request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        method = data.get('method')
        params = data.get('params', {})
        request_id = data.get('id')

        logging.info(f"HTTP MCP request: {method}")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "paper-search-server",
                        "version": "0.1.3"
                    }
                }
            }
        elif method == "notifications/initialized":
            return '', 204
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Primary search tool that returns object IDs for academic papers (OpenAI MCP standard)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of results to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "fetch",
                            "description": "Fetch detailed paper information using object ID (OpenAI MCP standard)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Object ID to retrieve"}
                                },
                                "required": ["id"]
                            }
                        },
                        {
                            "name": "search_cached",
                            "description": "Search through cached results for specific terms",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query_terms": {"type": "string", "description": "Terms to search for in cached results"},
                                    "limit": {"type": "integer", "description": "Maximum number of results to return (default: 5)", "default": 5}
                                },
                                "required": ["query_terms"]
                            }
                        },
                        {
                            "name": "search_arxiv",
                            "description": "Search academic papers from arXiv",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_pubmed",
                            "description": "Search academic papers from PubMed",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_biorxiv",
                            "description": "Search academic papers from bioRxiv",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_medrxiv",
                            "description": "Search academic papers from medRxiv",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_google_scholar",
                            "description": "Search academic papers from Google Scholar",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_iacr",
                            "description": "Search academic papers from IACR ePrint Archive",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10},
                                    "fetch_details": {"type": "boolean", "description": "Whether to fetch detailed information (default: true)", "default": True}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_semantic",
                            "description": "Search academic papers from Semantic Scholar",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "year": {"type": "string", "description": "Optional year filter (e.g., '2019', '2016-2020')"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "search_crossref",
                            "description": "Search academic papers from CrossRef database",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query string"},
                                    "max_results": {"type": "integer", "description": "Maximum number of papers to return (default: 10)", "default": 10}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "get_crossref_paper_by_doi",
                            "description": "Get a specific paper from CrossRef by its DOI",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "doi": {"type": "string", "description": "Digital Object Identifier"}
                                },
                                "required": ["doi"]
                            }
                        },
                        {
                            "name": "download_arxiv",
                            "description": "Download PDF of an arXiv paper",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "paper_id": {"type": "string", "description": "arXiv paper ID"},
                                    "save_path": {"type": "string", "description": "Directory to save the PDF (default: './downloads')", "default": "./downloads"}
                                },
                                "required": ["paper_id"]
                            }
                        },
                        {
                            "name": "read_arxiv_paper",
                            "description": "Read and extract text content from an arXiv paper PDF",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "paper_id": {"type": "string", "description": "arXiv paper ID"},
                                    "save_path": {"type": "string", "description": "Directory where the PDF is/will be saved (default: './downloads')", "default": "./downloads"}
                                },
                                "required": ["paper_id"]
                            }
                        }
                    ]
                }
            }
        elif method == "tools/call":
            tool_name = params.get('name')
            tool_args = params.get('arguments', {})
            result = None

            if tool_name == "search":
                result = asyncio.run(search(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "fetch":
                result = asyncio.run(fetch(tool_args.get('id')))
            elif tool_name == "search_cached":
                result = asyncio.run(search_cached(
                    tool_args.get('query_terms'),
                    tool_args.get('limit', 5)
                ))
            elif tool_name == "search_arxiv":
                result = asyncio.run(search_arxiv(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "search_pubmed":
                result = asyncio.run(search_pubmed(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "search_biorxiv":
                result = asyncio.run(search_biorxiv(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "search_medrxiv":
                result = asyncio.run(search_medrxiv(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "search_google_scholar":
                result = asyncio.run(search_google_scholar(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "search_iacr":
                result = asyncio.run(search_iacr(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10),
                    tool_args.get('fetch_details', True)
                ))
            elif tool_name == "search_semantic":
                result = asyncio.run(search_semantic(
                    tool_args.get('query'),
                    tool_args.get('year'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "search_crossref":
                result = asyncio.run(search_crossref(
                    tool_args.get('query'),
                    tool_args.get('max_results', 10)
                ))
            elif tool_name == "get_crossref_paper_by_doi":
                result = asyncio.run(get_crossref_paper_by_doi(tool_args.get('doi')))
            elif tool_name == "download_arxiv":
                result = asyncio.run(download_arxiv(
                    tool_args.get('paper_id'),
                    tool_args.get('save_path', './downloads')
                ))
            elif tool_name == "read_arxiv_paper":
                result = asyncio.run(read_arxiv_paper(
                    tool_args.get('paper_id'),
                    tool_args.get('save_path', './downloads')
                ))
            else:
                raise Exception(f"Unknown tool: {tool_name}")

            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
                }
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                }
            }

        return jsonify(response)

    except Exception as e:
        logging.error(f"HTTP MCP request error: {e}")
        return jsonify({
            "jsonrpc": "2.0",
            "id": request.get_json().get('id') if request.get_json() else None,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }), 500

if __name__ == "__main__":
    transport_type = os.environ.get("MCP_TRANSPORT", "stdio")
    port = int(os.environ.get("PORT", 3000))

    if transport_type == "http":
        logging.info(f"Starting Paper Search MCP server on HTTP port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logging.info("Starting Paper Search MCP server on stdio")
        mcp.run(transport="stdio")
