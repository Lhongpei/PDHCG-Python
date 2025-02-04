from typing import Optional
class Params:
    def __init__(self) -> None:
        self.gpu_flag = False #Whether using GPU
        self.warm_up_flag = False #Whether using warm-up desigened for exclude compilation time
        self.verbose_level = 2 #The level of verbosity, 0-9 (1: no output, 9: all output)
        self.time_limit = 3600.0# The time limit for the solver
        self.relat_error_tolerance = 1e-6 #The relative error tolerance
        self.iteration_limit = 2**31 - 1# The iteration limit
        self.ruiz_rescaling_iters = 10# The number of iterations for Ruiz rescaling
        self.l2_norm_rescaling_flag = False# Whether using L2 norm rescaling
        self.pock_chambolle_alpha = 1.0# The alpha parameter for Pock-Chambolle algorithm
        self.artificial_restart_threshold = 0.2# The threshold for artificial restart
        self.sufficient_reduction = 0.2# The sufficient reduction in judging restart
        self.necessary_reduction = 0.8# The necessary reduction in judging restart
        self.primal_weight_update_smoothing = 0.2 # The smoothing parameter for primal-dual weight update
        self.save_flag = False # Whether saving the result
        self.saved_name = None # The name for saving the result
        self.output_dir = None # The output directory for saving the result
        
    def setParam(self, Param: str, value: str, verbose: bool = False):
        if verbose:
            print(f"Setting {Param} to {value}")
        setattr(self, Param, value)
    
    def setParams(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)