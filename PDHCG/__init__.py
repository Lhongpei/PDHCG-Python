# __init__.py
import os
import sys
import logging
from juliacall import Main as jl  # 直接导入 jl

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_julia():
    """Safely initialize the Julia environment, calling Pkg.develop only when necessary."""
    try:
        # Import the Pkg module and activate the target environment.
        jl.seval('using Pkg')
        if not os.path.exists("pdhcg_core"):
            os.mkdir("pdhcg_core")
        jl.seval('Pkg.activate("pdhcg_core")')
        
        try:
            # Attempt to load the PDHCG package.
            # If it is already developed, this should succeed.
            jl.seval('using PDHCG')
        except Exception as load_err:
            # If loading fails, it means PDHCG is not registered in the current environment.
            logger.info("PDHCG not found in the current environment. Calling Pkg.develop to register the local package...")
            jl.seval('Pkg.develop(path="PDHCG")')
            # Reload the PDHCG package after developing it.
            jl.seval('using PDHCG')
        
        # Load additional packages if necessary.
        jl.seval('using SparseArrays')
        logger.info("Julia package PDHCG loaded successfully.")
        
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