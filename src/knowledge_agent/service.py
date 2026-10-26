from functools import lru_cache

from knowledge_agent.agent import KnowledgeAgent
from knowledge_agent.config import get_settings
from knowledge_agent.retrieval import KnowledgeIndex


@lru_cache
def get_index() -> KnowledgeIndex:
    return KnowledgeIndex(get_settings())


@lru_cache
def get_agent() -> KnowledgeAgent:
    return KnowledgeAgent(get_settings(), get_index())
