from gotranx.templates import cytozoo


def test_model_struct_has_one_parameters_field():
    code = cytozoo.model_struct("ORdmmLand", 177)
    assert "struct ORdmmLand{T <: AbstractVector} <: AbstractCardiacCellModel" in code
    assert "parameters::T" in code
    # ToRORd carries celltype and stim; a generated model must not invent them.
    assert "celltype" not in code
    assert "stim" not in code
    # Three constructors, matching ToRORd's shape.
    assert "ORdmmLand() = ORdmmLand(Float64)" in code
    assert "function ORdmmLand(::Type{ElT}) where {ElT <: Number}" in code
    assert "function ORdmmLand(::Type{VT}) where {VT <: AbstractVector}" in code


def test_index_lookups_raise_rather_than_return_minus_one():
    code = cytozoo.index_lookups("ORdmmLand", ["v", "cai"], ["GNa"])
    assert "state_index(::ORdmmLand, name::Symbol)" in code
    assert "parameter_index(::ORdmmLand, name::Symbol)" in code
    # A Dict lookup raises KeyError on an unknown name; -1 must appear nowhere.
    assert "-1" not in code
