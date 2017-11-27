# Data structures to represent the block and enlarged block objects.
immutable Block
    length::Int
    basis_size::Int
    operator_dict::Dict{Symbol,AbstractMatrix{Float64}}
end

immutable EnlargedBlock
    length::Int
    basis_size::Int
    operator_dict::Dict{Symbol,AbstractMatrix{Float64}}
end

# For these objects to be valid, the basis size must match the dimension of
# each operator matrix.
isvalid(block::Union{Block,EnlargedBlock}) =
    all(op -> size(op) == (block.basis_size, block.basis_size), values(block.operator_dict))

# Model-specific code for the Heisenberg XXZ chain
model_d = 2  # single-site basis size

Sz1 = [0.5 0.0; 0.0 -0.5]  # single-site S^z
Sp1 = [0.0 1.0; 0.0 0.0]  # single-site S^+

H1 = [0.0 0.0; 0.0 0.0]  # single-site portion of H is zero

function H2(Sz1, Sp1, Sz2, Sp2)  # two-site part of H
    # Given the operators S^z and S^+ on two sites in different Hilbert spaces
    # (e.g. two blocks), returns a Kronecker product representing the
    # corresponding two-site term in the Hamiltonian that joins the two sites.
    const J = 1.0
    const Jz = 1.0
    return (J / 2) * (kron(Sp1, Sp2') + kron(Sp1', Sp2)) + Jz * kron(Sz1, Sz2)
end

function enlarge_block(block::Block)
    # This function enlarges the provided Block by a single site, returning an
    # EnlargedBlock.
    mblock = block.basis_size
    o = block.operator_dict

    # Create the new operators for the enlarged block.  Our basis becomes a
    # Kronecker product of the Block basis and the single-site basis.  NOTE:
    # `kron` uses the tensor product convention making blocks of the second
    # array scaled by the first.  As such, we adopt this convention for
    # Kronecker products throughout the code.
    enlarged_operator_dict = Dict{Symbol,AbstractMatrix{Float64}}(
        :H => kron(o[:H], speye(model_d)) + kron(speye(mblock), H1) + H2(o[:conn_Sz], o[:conn_Sp], Sz1, Sp1),
        :conn_Sz => kron(speye(mblock), Sz1),
        :conn_Sp => kron(speye(mblock), Sp1),
    )

    return EnlargedBlock(block.length + 1,
                         block.basis_size * model_d,
                         enlarged_operator_dict)
end