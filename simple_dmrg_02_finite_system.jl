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
function enlarge_block_right(block::Block, H::NNHam)
    enlarged_block = Block(block.L+1, block.χ * H.p)
    for (sym, op) in H.opdict
        enlarged_block[sym] = kronr(speye(block.χ), op)
    end

    hL =  kronr(block[:H], speye(H.p))
    hLA = H2(H, block.opdict, H.opdict)
    hA = kronr(speye(block.χ), H1(H))
    enlarged_block[:H] = hL + hLA + hA 

    return enlarged_block
end

function enlarge_block_left(H::NNHam, block::Block)
    enlarged_block = Block(block.L+1, block.χ * H.p)
    for (sym, op) in H.opdict
        enlarged_block[sym] = kronr(op, speye(block.χ))
    end

    hB = kronr(H1(H), speye(block.χ))
    hBR = H2(H, H.opdict, block.opdict)
    hR =  kronr(speye(H.p), block[:H])
    enlarged_block[:H] = hB + hBR + hR

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
    blockLA = enlarge_block_right(blockL, H)
    blockBR = enlarge_block_left(H, blockR)

    @assert isvalid(blockLA)
    @assert isvalid(blockBR)

    # Construct the full superblock Hamiltonian on sites LABR. 
    @assert blockLA.χ == blockL.χ*H.p
    @assert blockBR.χ == blockR.χ*H.p

    H_LABR = kronr(blockLA[:H], speye(blockBR.χ)) + 
             kronr(speye(blockLA.χ), blockBR[:H]) + 
             H2(H, blockLA.opdict, blockBR.opdict)
    # Call ARPACK to find the superblock ground state.  (:SR means find the
    # eigenvalue with the "smallest real" value.)
    #
    # But first, we explicitly modify the matrix so that it will be detected as
    # Hermitian by `eigs`.  (Without this step, the matrix is effectively
    # Hermitian but won't be detected as such due to small roundoff error.)
    H_LABR = (H_LABR + H_LABR') / 2
    (energy,), psi0 = eigs(H_LABR, nev=1, which=:SR)
    # Construct the reduced density matrix of the system by tracing out the
    # environment
    psi0 = reshape(psi0, (blockLA.χ, blockBR.χ))
    U, s, Vd = svd(psi0)
    V = Vd'

    # Truncate using the `χ` overall most significant eigenvectors.
    χ, U, s, V = truncatesvd(U, s, V; kwargs...)
    sV = Diagonal(s)*V

    tensorA = reshape(U, blockL.χ, H.p, χ)
    tensorB = reshape(sV, χ, H.p, blockR.χ)

    # Rotate and truncate each operator.
    newblockLA = project(blockLA, U)
    newblockBR = project(blockBR, V')

    truncation_error = 1 - vecnorm(s)^2
    println("truncation error: ", truncation_error)
    return newblockLA, newblockBR, energy
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
    
    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # (:l) and right (:r) block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
    blockL = blockR = initial_block(H)
    block_disk[:l, 1] = blockL
    block_disk[:r, 1] = blockR
    while (blockL.L+blockR.L) < L
        # Perform a single DMRG step and save the new Block to "disk"
        println(graphic(blockL, blockR))
        blockL, blockR, energy = single_dmrg_step(H, blockL, blockR; χmax = χ_inf)
        println("E/L = ", energy / (blockL.L + blockR.L))
        block_disk[:l, blockL.L] = blockL
        block_disk[:r, blockR.L] = blockR
    end

    # Now that the system is built up to its full size, we perform sweeps using
    # the finite system algorithm.  At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
    loc = blockL.L
    dir=:R

    for χ in χ_sweep
        while true
            if dir==:R
                blockL = block_disk[:l, loc]
                blockR = block_disk[:r, L - loc - 2]
                println(repeat("=", blockL.L) * "**" * repeat("-", blockR.L))
                blockL, _, energy = single_dmrg_step(H, blockL, blockR; χmax=χ)
                block_disk[:l, loc+1] = blockL
                if loc >= L-1
                    dir = :L
                else
                    loc += 1
                end

            elseif dir==:L
                blockL = block_disk[:l, loc-2]
                blockR = block_disk[:r, L - loc]
                println(repeat("-", blockL.L) * "**" * repeat("=", blockR.L))
                _, blockR, energy = single_dmrg_step(H, blockL, blockR; χmax=χ)
                block_disk[:l, L-loc+1] = blockR
                if loc <=2
                    dir = :R
                else
                    loc -= 1
                end
            end

            println("E/L = ", energy / L)

            # Check whether we just completed a full sweep.
            if dir == :R && 2 * loc == L
                break  # escape from the "while true" loop
            end
        end
    end
end

σᶻ = [1.0 0.0; 0.0 -1.0]  # single-site S^z
σ⁺ = [0.0 2.0; 0.0 0.0]  # single-site S^+
σˣ = [0.0 1.0; 1.0 0.0]
iσʸ = [0.0 1.0; -1.0 0.0]

# HeisenbergXXZ(J::Float64, Jz::Float64) = NNHam(2, :Z, 0., [:Z, :P, :M], [:Z, :M, :P], [Jz/4, J/8, J/8], Dict(:Z=>σᶻ, :P=>σ⁺, :M=>σ⁺'))
HeisenbergXXZ(J::Float64, Jz::Float64) = NNHam(2, :Z, 0., [:Z, :X, :Y], [:Z, :X, :Y], [Jz/4, J/4, -J/4], Dict(:Z=>σᶻ, :X=>σˣ, :Y=>iσʸ))

H = HeisenbergXXZ(1.0, 1.0)
finite_system_algorithm(H, 20, 10, [10, 20, 30, 40, 40])
