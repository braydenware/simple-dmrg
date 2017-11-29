#!/usr/bin/env julia
#
# Simple DMRG tutorial.  This code integrates the following concepts:
#  - Infinite system algorithm
#  - Finite system algorithm
#
# Copyright 2013-2015 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/simple-dmrg/>

# Extended by Brayden Ware for learning purposes
# Do not distribute

kronr(args...) = length(args)==1 ? args[1] : kron(reverse(args)...)

# Data structures to represent the block and enlarged block objects.
immutable Block <: Associative{Symbol, AbstractMatrix{Float64}}
    L::Int
    χ::Int
    opdict::Dict{Symbol,AbstractMatrix{Float64}}
end
Block(L::Int, χ::Int) = Block(L, χ, Dict{Symbol, AbstractMatrix{Float64}}())
# Associative interface
Base.start(b::Block) = start(b.opdict)
Base.next(b::Block, state) = next(b.opdict, state)
Base.done(b::Block, state) = done(b.opdict, state)
Base.length(b::Block) = length(b.opdict)
Base.show(b::Block) = show(STDOUT, b)
Base.get(b::Block, s::Symbol, default) = get(b.opdict, s, default)
Base.similar(b::Block) = Block(b.L, b.χ)
# Free methods: keys, values, push!, merge!, in, haskey, getindex, filter, copy, ==, convert(::Dict{Symbol, AbstractMatrix{Float64}}, )

Base.setindex!(b::Block, v::AbstractMatrix{Float64}, s::Symbol) = b.opdict[s]=v

"""
Checks dimensions of block operators
"""
isvalid(block::Block) = all(op -> size(op) == (block.χ, block.χ), values(block))

struct NNHam
    p::Int

    H1::Symbol
    H1coeff::Float64
    
    H2left::Vector{Symbol}
    H2right::Vector{Symbol}
    H2coeff::Vector{Float64}
    
    opdict::Dict{Symbol, AbstractMatrix{Float64}}
end

σᶻ = [0.5 0.0; 0.0 -0.5]  # single-site S^z
σ⁺ = [0.0 1.0; 0.0 0.0]  # single-site S^+

HeisenbergXXZ(J::Float64, Jz::Float64) = NNHam(2, :Z, 0., [:Z, :P, :M], [:Z, :M, :P], [Jz, J/2, J/2], 
                                               Dict(:Z=>σᶻ, :P=>σ⁺, :M=>σ⁺'))

function H2(H::NNHam, leftopdict::Dict{Symbol, AbstractMatrix{Float64}}, rightopdict::Dict{Symbol, AbstractMatrix{Float64}})
    ans = sum(c*kronr(leftopdict[Hl],rightopdict[Hr]) for (Hl, Hr, c) in zip(H.H2left, H.H2right, H.H2coeff))
    return ans
end
H2(H::NNHam) = H2(H, H.opdict, H.opdict)

H1(H::NNHam, opdict::Dict{Symbol, AbstractMatrix{Float64}}) = H.H1coeff*opdict[H.H1]
H1(H::NNHam) = H1(H, H.opdict)

function initial_block(H::NNHam)
    block = Block(1, H.p, H.opdict) # block knows how to connnect left or right using H2left or H2right symbols
    block[:H] = H1(H)
    return block
end

"""
This function enlarges the provided Block by a single site.

Create the new operators for the enlarged block.  Our basis becomes a
Kronecker product of the Block basis and the single-site basis.  NOTE:
`kronr` uses the tensor product convention making blocks of the first
array scaled by the second.  As such, we adopt this convention for
Kronecker products throughout the code.
"""
function enlarge_block(H::NNHam, block::Block)
    enlarged_block = Block(block.L+1, block.χ * H.p)
    for (sym, op) in H.opdict
        enlarged_block[sym] = kronr(speye(block.χ), op)
    end

    hL =  kronr(block[:H], speye(H.p))
    hA = kronr(speye(block.χ), H1(H))
    hLA = H2(H, block.opdict, H.opdict)
    enlarged_block[:H] = hL + hA + hLA

    return enlarged_block
end

function project(operator::AbstractMatrix{Float64}, basis::AbstractMatrix{Float64})
    return basis' * (operator * basis)
end

function project(block::Block, basis::AbstractMatrix{Float64})
    newblock = Block(block.L, size(basis, 2))
    for (sym, op) in block
        newblock[sym] = project(op, basis)
    end
    return newblock
end

function truncatesvd(U, s, V; kwargs...)
    kwargs = Dict(kwargs)

    χ = length(s)
    if :smin in keys(kwargs)
        while s[χ]<kwargs[:smin] && χ>1
            χ -= 1
        end
    elseif :tol in keys(kwargs)
        while vecnorm(s[χ+1:end])^2<kwargs[:tol] && χ>1
            χ -= 1
        end 
    end

    if :χmax in keys(kwargs)
        χ = min(χ, kwargs[:χmax])
    end

    U = U[:, 1:χ]
    s = s[1:χ]
    V = V[1:χ, :]
    return χ, U, s, V
end

"""
Performs a single DMRG step using `left` as the system and `right` as the
environment, keeping a maximum of `χmax` states in the new basis.
"""
function single_dmrg_step(H::NNHam, blockL::Block, blockR::Block; kwargs...)
    @assert isvalid(blockL)
    @assert isvalid(blockR)

    # Enlarge each block by a single site.
    blockLA = enlarge_block(H, blockL)
    if blockL === blockR  # no need to recalculate a second time
        blockRB = blockLA
    else
        blockRB = enlarge_block(H, blockR)
    end

    @assert isvalid(blockLA)
    @assert isvalid(blockRB)

    # Construct the full superblock Hamiltonian.
    @assert blockLA.χ == blockL.χ*H.p
    @assert blockRB.χ == blockR.χ*H.p

    superblock_hamiltonian = kronr(blockLA[:H], speye(blockRB.χ)) + kronr(speye(blockLA.χ), blockRB[:H]) + H2(H, blockLA.opdict, blockRB.opdict)
    # Call ARPACK to find the superblock ground state.  (:SR means find the
    # eigenvalue with the "smallest real" value.)
    #
    # But first, we explicitly modify the matrix so that it will be detected as
    # Hermitian by `eigs`.  (Without this step, the matrix is effectively
    # Hermitian but won't be detected as such due to small roundoff error.)
    superblock_hamiltonian = (superblock_hamiltonian + superblock_hamiltonian') / 2
    (energy,), psi0 = eigs(superblock_hamiltonian, nev=1, which=:SR)
    # Construct the reduced density matrix of the system by tracing out the
    # environment
    psi0 = reshape(psi0, (blockLA.χ, blockRB.χ))
    U, s, Vd = svd(psi0)
    V = Vd'

    # Truncate using the `χ` overall most significant eigenvectors.
    χ, U, s, V = truncatesvd(U, s, V; kwargs...)
    sV = Diagonal(s)*V

    tensorA = reshape(U, blockL.χ, H.p, χ)
    tensorB = permutedims(reshape(sV, χ, blockR.χ, H.p), (1, 3, 2))

    # Rotate and truncate each operator.
    newblockLA = project(blockLA, U)

    truncation_error = 1 - vecnorm(s)^2
    println("truncation error: ", truncation_error)
    return newblockLA, energy
end

function graphic(sys_block::Block, env_block::Block, sys_label::Symbol=:l)
    # Returns a graphical representation of the DMRG step we are about to
    # perform, using '=' to represent the system sites, '-' to represent the
    # environment sites, and '**' to represent the two intermediate sites.
    str = repeat("=", sys_block.L) * "**" * repeat("-", env_block.L)
    if sys_label == :r
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        str = reverse(str)
    elseif sys_label != :l
        throw(ArgumentError("sys_label must be :l or :r"))
    end
    return str
end

function infinite_system_algorithm(H::NNHam, L::Int, χ::Int)
    block = initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.L < L
        println("L = ", block.L * 2 + 2)
        block, energy = single_dmrg_step(H, block, block; χmax=χ)
        println("E/L = ", energy / (block.L * 2))
    end
end

function finite_system_algorithm(H::NNHam, L::Int, χ_inf::Int, χ_sweep::AbstractVector{Int})
    @assert iseven(L)

    # To keep things simple, this dictionary is not actually saved to disk, but
    # we use it to represent persistent storage.
    block_disk = Dict{Tuple{Symbol,Int},Block}()  # "disk" storage for Block objects
    site_tensors_disk = Dict{Tuple{Symbol, Int}, Array{Float64, 3}}()

    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # (:l) and right (:r) block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
    block = initial_block(H)
    block_disk[:l, block.L] = block
    block_disk[:r, block.L] = block
    while 2 * block.L < L
        # Perform a single DMRG step and save the new Block to "disk"
        println(graphic(block, block))
        block, energy = single_dmrg_step(H, block, block; χmax = χ_inf)
        println("E/L = ", energy / (block.L * 2))
        block_disk[:l, block.L] = block
        block_disk[:r, block.L] = block
    end

    # Now that the system is built up to its full size, we perform sweeps using
    # the finite system algorithm.  At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
    sys_label, env_label = :l, :r

    # Rename block -> sys_block
    sys_block = block
    block = Block(0, 0)

    for χ in χ_sweep
        while true
            # Load the appropriate environment block from "disk"
            env_block = block_disk[env_label, L - sys_block.L - 2]
            if env_block.L == 1
                # We've come to the end of the chain, so we reverse course.
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label
            end

            # Perform a single DMRG step.
            println(graphic(sys_block, env_block, sys_label))
            sys_block, energy = single_dmrg_step(H, sys_block, env_block; χmax=χ)

            println("E/L = ", energy / L)

            # Save the block from this step to disk.
            block_disk[sys_label, sys_block.L] = sys_block

            # Check whether we just completed a full sweep.
            if sys_label == :l && 2 * sys_block.L == L
                break  # escape from the "while true" loop
            end
        end
    end
end

#infinite_system_algorithm(100, 20)
H = HeisenbergXXZ(1.0, 1.0)
finite_system_algorithm(H, 20, 10, [10, 20, 30, 40, 40])
