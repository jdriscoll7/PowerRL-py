from rlpower.power.powermodels_interface import load_test_case, solve_opf

network = load_test_case()
solved_network = solve_opf(network)