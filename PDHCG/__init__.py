# __init__.py
import os
import sys
import logging
from juliacall import Main as jl  # 直接导入 jl

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_julia():
    """安全初始化 Julia 环境"""
    try:
        jl.seval(f'using Pkg')
        #print all LOAD_PATH
        jl.seval(f'Pkg.activate("PDHCG")')
        jl.seval(f'push!(LOAD_PATH, "PDHCG/src")')
        jl.seval('using PDHCG')
        jl.seval('using SparseArrays')
        logger.info("Julia module PDHCG loaded successfully.")
        
        # jl.seval('using SparseArrays')
        # logger.info("Julia module SparseArrays loaded successfully.")

    except Exception as e:
        logger.error(f"Julia initialization failed: {e}")
        raise RuntimeError("Julia environment initialization failed.") from e

# 可选：自动初始化（根据需求开启）
init_julia()

# 导入 Python 模块（确保在 Julia 初始化后）

from .params import Params
from .probs import QuadraticProgrammingProblem, read_problem, generate_problem
from .solver import PDHCG



# 暴露公共接口
__all__ = [
    "PDHCG",
    "QuadraticProgrammingProblem",
    "read_problem",
    "generate_problem",
    "Params",
    "jl",  # 全局 Julia 接口
    "init_julia",  # 初始化函数
]