"""
pytest 配置和夹具
"""

import pytest
from fastapi.testclient import TestClient

from src.com.mx.core.app import create_app


@pytest.fixture
def app():
    """创建测试应用"""
    app = create_app()
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)