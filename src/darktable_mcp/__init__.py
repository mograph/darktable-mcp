from .server import mcp


def main():
    mcp.run()


__all__ = ["mcp", "main"]
