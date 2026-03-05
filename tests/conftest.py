"""Shared pytest fixtures."""
import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default asyncio event loop policy for tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
