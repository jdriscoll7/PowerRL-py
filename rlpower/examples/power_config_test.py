from rlpower.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager

if __name__ == "__main__":
    network = load_test_case()
    config = Configuration(network)
    config_manager = ConfigurationManager(network)
    # config_manager.apply_network_configuration(123)
    config_manager.solve_branch_configuration(binary_configuration=0b111, branch_id="3")
    config_manager.get_branch_state("3")

    config_manager.solve_configuration()
