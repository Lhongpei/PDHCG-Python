# compile.jl
using PackageCompiler
using PDHCG

# precompile.jl
open("precompile.jl", "w") do f
    write(f, """
    using PDHCG
    PDHCG.pdhcgSolve()
    PDHCG.generateProblem()
    """)
end


create_sysimage(
    ["PDHCG"],
    sysimage_path="PDHCG_sysimage.so",
    precompile_execution_file="precompile.jl"
)