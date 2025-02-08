import subprocess
import sys
def check_julia_installed():
    """Check if Julia is installed and return its version."""
    try:
        julia_version = subprocess.check_output(["julia", "--version"], stderr=subprocess.STDOUT).decode().strip()
        print(f"✅ Julia detected: {julia_version}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Julia is not detected. Please install Julia first: https://julialang.org/downloads/")
        return False

def logoInit():
    if not check_julia_installed():
        sys.exit("❌ Installation failed: Julia is not installed. Please install Julia first.")
    print(logo)
    print("🚀 PDHCG has been successfully installed. If you have any problem, please contact ishongpeili@gmail.com.")

logo = r"""
    ██████╗ ██████╗ ██╗  ██╗ ██████╗ ██████╗ 
    ██╔══██╗██╔══██╗██║  ██║██╔════╝██╔════╝ 
    ██████╔╝██║  ██║███████║██║     ██║  ███╗
    ██╔═══╝ ██║  ██║██╔══██║██║     ██║   ██║
    ██║     ██████╔╝██║  ██║╚██████╗╚██████╔╝
    ╚═╝     ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ 
        An optimizer for Large Convex Quadratic Programming            
    """