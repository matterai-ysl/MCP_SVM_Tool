"""Entry point for running the MCP Neural Network Tool server."""

import asyncio

def main():
    """Main entry point for the package."""
    from .mcp_server import main as server_main
    asyncio.run(server_main())

if __name__ == "__main__":
    main()
