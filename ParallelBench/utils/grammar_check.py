
import os
from pathlib import Path

language_tool = None


def get_language_tool():
    global language_tool
    if language_tool is None:
        import language_tool_python

        # Check if LanguageTool is already cached to avoid re-downloading
        cache_dir = Path.home() / ".cache" / "language_tool_python"

        # If cache exists, use remote_server mode or disable auto-download
        if cache_dir.exists() and any(cache_dir.iterdir()):
            print(f"Using cached LanguageTool from {cache_dir}")
            # Find the cached version directory
            cached_versions = list(cache_dir.glob("LanguageTool-*"))
            if cached_versions:
                # Use the first cached version (offline mode)
                os.environ['LT_PATH'] = str(cached_versions[0])

        language_tool = language_tool_python.LanguageTool('en-US', config={"maxCheckThreads": 1, "maxSpellingSuggestions": 1})
    return language_tool


def grammar_check(text):
    text = text.strip()

    if text == "":
        return False

    tool = get_language_tool()
    matches = tool.check(text)
    return len(matches) == 0


def grammar_error_count(text):
    """
    Returns the number of grammar errors in the text.
    Lower count means better grammar.

    Args:
        text: Input text to check

    Returns:
        Number of grammar errors (0 means perfect grammar)
    """
    text = text.strip()

    if text == "":
        return float('inf')  # Heavily penalize empty text

    tool = get_language_tool()
    matches = tool.check(text)
    return len(matches)