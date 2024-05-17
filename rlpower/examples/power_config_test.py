from rlpower.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager

if __name__ == "__main__":
    network = load_test_case()
    config = Configuration(network)
    config_manager = ConfigurationManager(network)
    print("")
