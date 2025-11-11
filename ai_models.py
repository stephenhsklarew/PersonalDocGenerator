"""
AI Model integration for multiple providers
Supports: Qwen (via Ollama), Anthropic Claude, OpenAI GPT, Google Gemini
"""

import os
import sys
from typing import Optional
from pathlib import Path

# Add UnifiedLLMClient to path
unified_llm_path = Path.home() / 'Development' / 'Scripts' / 'UnifiedLLMClient'
sys.path.insert(0, str(unified_llm_path))

from llm_client import get_client


class AIModelManager:
    """Manages different AI models from various providers"""

    # Available models
    MODELS = {
        # Qwen models (via Ollama - local, free)
        'qwen-32b': {'provider': 'qwen', 'model': 'qwen2.5:32b', 'name': 'Qwen 2.5 32B (Local, Free, Default)'},
        'qwen-72b': {'provider': 'qwen', 'model': 'qwen2.5:72b', 'name': 'Qwen 2.5 72B (Local, Free, Most Capable)'},
        'qwen-14b': {'provider': 'qwen', 'model': 'qwen2.5:14b', 'name': 'Qwen 2.5 14B (Local, Free)'},
        'qwen-7b': {'provider': 'qwen', 'model': 'qwen2.5:7b', 'name': 'Qwen 2.5 7B (Local, Free, Fastest)'},

        # Anthropic Claude models
        'claude-3-5-sonnet': {'provider': 'claude', 'model': 'claude-3-5-sonnet-20241022', 'name': 'Claude 3.5 Sonnet'},
        'claude-3-5-haiku': {'provider': 'claude', 'model': 'claude-3-5-haiku-20241022', 'name': 'Claude 3.5 Haiku (Fast)'},
        'claude-3-opus': {'provider': 'claude', 'model': 'claude-3-opus-20240229', 'name': 'Claude 3 Opus (Most Capable)'},

        # OpenAI GPT models
        'gpt-4': {'provider': 'openai', 'model': 'gpt-4-turbo-preview', 'name': 'GPT-4 Turbo'},
        'gpt-4o': {'provider': 'openai', 'model': 'gpt-4o', 'name': 'GPT-4o (Multimodal)'},
        'gpt-3.5-turbo': {'provider': 'openai', 'model': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo (Fast)'},

        # Google Gemini models
        'gemini-pro': {'provider': 'gemini', 'model': 'gemini-pro', 'name': 'Gemini Pro'},
        'gemini-2.5-flash': {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'name': 'Gemini 2.5 Flash (Fast)'},
        'gemini-2.5-pro': {'provider': 'gemini', 'model': 'gemini-2.5-pro', 'name': 'Gemini 2.5 Pro'},
        'gemini-1.5-pro': {'provider': 'gemini', 'model': 'gemini-1.5-pro-latest', 'name': 'Gemini 1.5 Pro'},
        'gemini-1.5-flash': {'provider': 'gemini', 'model': 'gemini-1.5-flash-latest', 'name': 'Gemini 1.5 Flash'},
    }

    def __init__(self, model_key: str = 'qwen-32b'):
        """Initialize AI Model Manager"""
        self.model_key = model_key
        self.model_info = self.MODELS.get(model_key)

        if not self.model_info:
            raise ValueError(f"Unknown model: {model_key}")

        self.provider = self.model_info['provider']
        self.model = self.model_info['model']
        self.name = self.model_info['name']

        # Initialize UnifiedLLMClient
        self._init_client()

    def _init_client(self):
        """Initialize UnifiedLLMClient with the appropriate provider"""
        try:
            self.client = get_client(provider=self.provider)
        except Exception as e:
            raise ValueError(f"Failed to initialize {self.provider} client: {str(e)}")

    def generate(self, prompt: str, max_tokens: int = 16000) -> str:
        """Generate content using UnifiedLLMClient"""
        try:
            return self.client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
        except Exception as e:
            raise Exception(f"Error generating with {self.name}: {str(e)}")

    @classmethod
    def get_available_models(cls) -> dict:
        """Get list of available models"""
        return cls.MODELS

    @classmethod
    def list_models_by_provider(cls) -> dict:
        """List models grouped by provider"""
        providers = {}
        for key, info in cls.MODELS.items():
            provider = info['provider']
            if provider not in providers:
                providers[provider] = []
            providers[provider].append({
                'key': key,
                'name': info['name'],
                'model': info['model']
            })
        return providers

    @classmethod
    def get_model_display_list(cls) -> list:
        """Get formatted list for display in UI"""
        models = []
        by_provider = cls.list_models_by_provider()

        for provider in ['qwen', 'claude', 'openai', 'gemini']:
            if provider in by_provider:
                provider_name = {
                    'qwen': 'Qwen (Local, Free)',
                    'claude': 'Anthropic Claude',
                    'openai': 'OpenAI GPT',
                    'gemini': 'Google Gemini'
                }[provider]

                models.append(f"\n{provider_name}:")
                for model in by_provider[provider]:
                    models.append(f"  • {model['key']}: {model['name']}")

        return models
