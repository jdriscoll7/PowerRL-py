from julia import Main
import copy
Main.include("./rlpower/power/pm_functions.jl")


class Configuration:
    def __init__(self, network: dict):
        self.config_dict = {"gen": {x: 0 for x in network["gen"].keys()},
                            "load": {x: 0 for x in network["load"].keys()},
                            "branch_to": {x: 0 for x in network["branch"].keys()},
                            "branch_from": {x: 0 for x in network["branch"].keys()},
                            "branch_to_on": {x: 1 for x in network["branch"].keys()},
                            "branch_from_on": {x: 1 for x in network["branch"].keys()}}

    def apply_binary(self, binary_config: int):

        binary_string = "{0:b}".format(binary_config)
        string_loc = 0

        for key in self.config_dict:
            for key_2 in self.config_dict[key]:
                self.config_dict[key][key_2] = binary_string[string_loc]
                string_loc += 1

    def reset(self):
        for key in self.config_dict:
            for key_2 in self.config_dict[key]:
                if key_2 == "branch_to_on" or key_2 == "branch_from_on":
                    self.config_dict[key][key_2] = 1
                else:
                    self.config_dict[key][key_2] = 0


class ConfigurationManager:
    def __init__(self, network: dict = None, path: str = None,):

        assert path is not None or network is not None

        # If network is provided, make sure it stores opt solution.
        if network is not None:
            self.network = Main.make_busbar_network(network)

        # If path is provided, load network and solve.
        if path is not None:
            loaded_network = load_test_case(path)
            self.network = Main.make_busbar_network(loaded_network)

        # In both cases, store an extra network that is used as reconfigured version of main network.
        self.configured_network = Main.make_busbar_network(network)

        # Load base configuration that will be used later.
        self.configuration = Configuration(self.network)

        # Store solutions for base network and configured network.
        self.solution = Main.pm_solve_opf(self.network)
        self.config_solution = Main.pm_solve_opf(self.configured_network)

    def apply_configuration(self, binary_configuration: int):

        # Update configuration.
        self.configuration.apply_binary(binary_configuration)

        # Modify configured network to reflect this configuration.


def solve_opf(network):
    return Main.pm_solve_opf(network)


def load_test_case(path=None):

    if path is None:
        path = 'ieee_data\\pglib_opf_case57_ieee.m'

    return Main.load_test_case(path)