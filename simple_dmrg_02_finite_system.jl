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
function enlarge_block(block::Block, H::NNHam)
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

function enlarge_block(H::NNHam, block::Block)
    enlarged_block = Block(block.L+1, block.χ * H.p)
    for (sym, op) in H.opdict
        enlarged_block[sym] = kronr(op, speye(block.χ))
    end

    hR =  kronr(speye(H.p), block[:H])
    hB = kronr(H1(H), speye(block.χ))
    hBR = H2(H, H.opdict, block.opdict)
    enlarged_block[:H] = hB + hR + hBR

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
    blockLA = enlarge_block(blockL, H)
    blockBR = enlarge_block(H, blockR)

    @assert isvalid(blockLA)
    @assert isvalid(blockBR)

    # Construct the full superblock Hamiltonian.
    @assert blockLA.χ == blockL.χ*H.p
    @assert blockBR.χ == H.p*blockR.χ

    superblock_hamiltonian = kronr(blockLA[:H], speye(blockBR.χ)) + kronr(speye(blockLA.χ), blockBR[:H]) + H2(H, blockLA.opdict, blockBR.opdict)
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
    psi0 = reshape(psi0, (blockLA.χ, blockBR.χ))
    U, s, Vd = svd(psi0)
    V = Vd'

    # Truncate using the `χ` overall most significant eigenvectors.
    χ, U, s, V = truncatesvd(U, s, V; kwargs...)
    
    # sV = Diagonal(s)*V
    # tensorA = reshape(U, blockL.χ, H.p, χ)
    # tensorB = reshape(sV, χ, H.p, blockR.χ)

    # Rotate and truncate each operator.
    newblockLA = project(blockLA, U)
    newblockBR = project(blockBR, V.')

    truncation_error = 1 - vecnorm(s)^2
    println("truncation error: ", truncation_error)
    return newblockLA, newblockBR, energy
end

struct Sweep
    L::Int
    l::UnitRange{Int}
    a::Int
    b::Int
    r::UnitRange{Int}
    Δ::Int
    firststep::Bool
end

Sweep(L, a, Δ, firststep=true) = Sweep(L, 1:a-1, a, a+1, a+2:L, Δ, firststep)

Base.start(sweep::Sweep) = sweep

function Base.next(sweep::Sweep, state::Sweep)    
    if state.Δ==1 && length(state.r)==1 || state.Δ==-1 && length(state.l)==1
        nextstate = Sweep(state.L, state.a-state.Δ, -state.Δ, false) # change directions and move in direction -Δ
    else
        nextstate = Sweep(sweep.L, state.a+state.Δ, state.Δ, false) # move in direction Δ
    end
    return state, nextstate
end

function Base.done(sweep::Sweep, state::Sweep)
    !state.firststep && sweep.a == state.a && sweep.Δ == state.Δ
end

"""
Returns a graphical representation of the DMRG step we are about to
perform, using '=' to represent the system sites, '-' to represent the
environment sites, and '**' to represent the two intermediate sites.
"""
function Base.show(io::IO, state::Sweep)
    str = fill('?', state.L)
    for i in 1:state.L
        if i in state.l
            str[i] = state.Δ==1 ? '=' : '-'
        elseif i==state.a
            str[i] = '*'
        elseif i==state.b
            str[i] ='*'
        elseif i in state.r
            str[i] = state.Δ==-1 ? '=' : '-'
        end
    end
    print(String(str))
end

function infinite_system_algorithm(H::NNHam, L::Int, χ::Int)
    leftblocks = Vector{Block}(L)
    rightblocks = Vector{Block}(L)

    leftblocks[1] = initial_block(H)
    rightblocks[1] = initial_block(H)
    sz = 1
    while 2*sz < L
        println("="^sz*"**"*"="^sz)
        leftblocks[sz+1], rightblocks[sz+1], energy = single_dmrg_step(H, leftblocks[sz], rightblocks[sz]; χmax = χ)
        println("E/L = ", energy / (sz * 2 + 2))
        sz+=1
    end
    println()
    return leftblocks, rightblocks
end

function sweep(H::NNHam, L::Int, initsite::Int, left_blocks::Vector{Block}, right_blocks::Vector{Block}, χ::Int)
    # At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
    energy = 0.
    for update in Sweep(L, initsite, +1)
        # Load the appropriate blocks from "disk"
        left_block = left_blocks[length(update.l)]
        right_block = right_blocks[length(update.r)]

        # Perform a single DMRG step.
        println(update)
        left_block, right_block, energy = single_dmrg_step(H, left_block, right_block; χmax=χ)

        println("E/L = ", energy / L)

        # Save the block from this step to disk.
        if update.Δ == 1
            left_blocks[length(update.l)+1] = left_block
        elseif update.Δ == -1
            right_blocks[length(update.r)+1] = right_block
        end
    end
    return left_blocks, right_blocks, energy
end

function finite_system_algorithm(H::NNHam, L::Int, χ_inf::Int, χ_sweep::AbstractVector{Int})
    # Use the infinite system algorithm to build up to desired size.
    leftblocks, rightblocks = infinite_system_algorithm(H, L, χ_inf)

    # Now that the system is built up to its full size, we perform sweeps.
    initsite = div(L, 2) + 1
    for (c, χ) in enumerate(χ_sweep)
        et = @elapsed leftblocks, rightblocks, energy = sweep(H, L, initsite, leftblocks, rightblocks, χ)
        println("Sweep $c with χ=$χ in time $et: energy/L=$(energy/L)")
        println()
    end
end

#infinite_system_algorithm(100, 20)
H = HeisenbergXXZ(1.0, 1.0)
finite_system_algorithm(H, 20, 10, [10, 20, 30, 40, 40])
