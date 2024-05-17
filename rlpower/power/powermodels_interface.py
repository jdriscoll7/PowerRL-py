from julia import Main
Main.include("./rlpower/power/pm_functions.jl")


def solve_opf(network):
    return Main.pm_solve_opf(network)


def load_test_case(path=None):

    if path is None:
        path = 'ieee_data\\pglib_opf_case57_ieee.m'

    return Main.load_test_case(path)