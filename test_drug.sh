#!/bin/bash
# 测试药物知识库系统

cd "$(dirname "$0")"
source venv/bin/activate
python test_drug_knowledge.py

