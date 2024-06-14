import pickle

import pandas as pd
import copy

# from julia import Julia
# Julia(init_julia=False)
from juliacall import Main, DictValue, VectorValue

Main.include("./rlpower/power/pm_functions.jl")


class ConfigurationDelta:
    def __init__(self, _id: int, _type: str, value: int, branch_config: str = None):
        self.id = _id
        self.type = _type
        self.branch_config = branch_config
        self.value = value


class Configuration:
    def __init__(self, network: dict, contains_busbars: bool = True):
        # self.config_dict = {"gen": {x: 0 for x in network["gen"].keys()},
        #                     "load": {x: 0 for x in network["load"].keys()},
        #                     "branch_to": {x: 0 for x in network["branch"].keys()},
        #                     "branch_from": {x: 0 for x in network["branch"].keys()},
        #                     "branch_to_on": {x: 1 for x in network["branch"].keys()},
        #                     "branch_from_on": {x: 1 for x in network["branch"].keys()}}

        self.branch_config = pd.DataFrame(0,
                                          index=[int(v) for v in network["branch"].keys()],
                                          columns=["inactive", "to_bus", "from_bus"])

        self.gen_config = pd.DataFrame(0,
                                       index=[int(v) for v in network["gen"].keys()],
                                       columns=["bus"])

        self.load_config = pd.DataFrame(0,
                                        index=[int(v) for v in network["load"].keys()],
                                        columns=["bus"])

        self.bus_config = pd.DataFrame(0,
                                       index=[int(v) for v in network["bus"].keys()],
                                       columns=["connected"])

        # Rename index for all three config dataframes.
        self.branch_config.index.name = "branch"
        self.gen_config.index.name = "gen"
        self.load_config.index.name = "load"

        self.n_gen = len(network["gen"])
        self.n_load = len(network["load"])
        self.n_branch = len(network["branch"])
        self.n_bus = len(network["bus"]) // 2 if contains_busbars else len(network["bus"])
        self.contains_busbars = contains_busbars

        self.pending_changes = []

        self.config_length = self.n_gen + self.n_load + 4 * self.n_branch
        self.config_length += self.n_bus if contains_busbars else 0

    def process_binary(self, binary_config: int):

        total_length = self.config_length
        assert binary_config < (1 << total_length)
        binary_string = f"{binary_config:0{total_length}b}"
        string_loc = 0

        # Set branch configurations.
        for config_key in self.branch_config.columns:
            branch_key = 1
            for i in range(self.n_branch):
                if self.branch_config.loc[branch_key, config_key] != int(binary_string[string_loc]):
                    self.branch_config.loc[branch_key, config_key] = int(binary_string[string_loc])
                    self.pending_changes.append(ConfigurationDelta(_id=branch_key,
                                                                   _type="branch",
                                                                   value=int(binary_string[string_loc]),
                                                                   branch_config=config_key))
                branch_key += 1
                string_loc += 1

        # Set load configurations.
        load_key = 1
        for i in range(self.n_load):
            if self.load_config.loc[load_key] != int(binary_string[string_loc]):
                self.load_config.loc[load_key] = int(binary_string[string_loc])
                self.pending_changes.append(ConfigurationDelta(_id=load_key,
                                                               _type="load",
                                                               value=int(binary_string[string_loc])))
            load_key += 1
            string_loc += 1

        # Set generator configurations.
        gen_key = 1
        for i in range(self.n_gen):
            if self.gen_config.loc[gen_key] != int(binary_string[string_loc]):
                self.gen_config.loc[gen_key] = int(binary_string[string_loc])
                self.pending_changes.append(ConfigurationDelta(_id=gen_key,
                                                               _type="gen",
                                                               value=int(binary_string[string_loc])))
            gen_key += 1
            string_loc += 1

        bus_key = 1
        for i in range(self.n_bus):
            if self.bus_config.loc[bus_key] != int(binary_string[string_loc]):
                self.bus_config.loc[bus_key] = int(binary_string[string_loc])
                self.pending_changes.append(ConfigurationDelta(_id=bus_key,
                                                               _type="bus",
                                                               value=int(binary_string[string_loc])))
            bus_key += 1
            string_loc += 1

    def process_branch_binary(self, binary_config: int, branch_id: str):

        assert binary_config < 8

        binary_string = f"{binary_config:0{3}b}"

        for i, b in enumerate(binary_string):
            config_key = self.branch_config.columns[i]
            if self.branch_config.loc[int(branch_id), config_key] != int(b):
                self.branch_config.loc[int(branch_id), config_key] = int(b)
                self.pending_changes.append(ConfigurationDelta(_id=int(branch_id),
                                                               _type="branch",
                                                               value=int(b),
                                                               branch_config=config_key))

    def apply_bus_binary(self, binary_config: int, bus_id: str):

        assert binary_config < 2

        if self.bus_config.loc[int(bus_id)] != binary_config:
            self.bus_config.loc[int(bus_id)] = binary_config
            self.pending_changes.append(ConfigurationDelta(_id=int(bus_id),
                                                           _type="bus",
                                                           value=int(binary_config)))

    def apply_gen_binary(self, binary_config: int, gen_id: str):

        assert binary_config < 2

        if self.gen_config.loc[int(gen_id)] != binary_config:
            self.gen_config.loc[int(gen_id)] = binary_config
            self.pending_changes.append(ConfigurationDelta(_id=int(gen_id),
                                                           _type="gen",
                                                           value=int(binary_config)))

    def apply_load_binary(self, binary_config: int, load_id: str):

        assert binary_config < 2

        if self.load_config.loc[int(load_id)] != binary_config:
            self.load_config.loc[int(load_id)] = binary_config
            self.pending_changes.append(ConfigurationDelta(_id=int(load_id),
                                                           _type="load",
                                                           value=int(binary_config)))

    def reset(self):

        for col in self.branch_config.columns:
            self.branch_config[col].values[:] = 0

        for col in self.load_config.columns:
            self.load_config[col].values[:] = 0

        for col in self.gen_config.columns:
            self.gen_config[col].values[:] = 0

        for col in self.bus_config.columns:
            self.bus_config[col].values[:] = 0

    def get_changes(self) -> list:
        return self.pending_changes

    def clear_changes(self) -> None:
        self.pending_changes = []


class ConfigurationManager:
    def __init__(self, network: dict = None, path: str = None, contains_busbars: bool = False):

        assert path is not None or network is not None

        self.contains_busbars = contains_busbars

        # If network is provided, make sure it stores opt solution.
        if network is not None:
            self.network = Main.make_busbar_network(network)

        # If path is provided, load network and solve.
        if path is not None:
            loaded_network = load_test_case(path)
            self.network = Main.make_busbar_network(loaded_network)

        # In both cases, store an extra network that is used as reconfigured version of main network.
        self.configured_network = copy.deepcopy(network)

        # Load base configuration that will be used later.
        self.configuration = Configuration(self.network, contains_busbars=contains_busbars)

        # Store solutions for base network and configured network.
        self.solution = solve_opf(self.network)
        self.config_solution = solve_opf(self.configured_network)

        self.branch_state_length = len(self.get_branch_state("1"))

    def get_branch_state(self, branch: str):

        n_branches = len(self.network["branch"])

        # Get existing branch opt data to know how many zeros to fill in if branch is inactive.

        # Get easy parameter and standard branch data.
        parameter_data = pd.to_numeric(pd.Series(self.network["branch"][branch]), errors="coerce").dropna()
        # print(self.config_solution["solution"].keys())
        if branch in self.config_solution["solution"]["branch"].keys():
            _branch_opt_data = self.config_solution["solution"]["branch"][branch]
        else:
            existing_key = list(self.config_solution["solution"]["branch"].keys())[0]
            _branch_opt_data = {k: 0 for k, v in self.config_solution["solution"]["branch"][existing_key].items()}

        to_bus = str(self.configured_network["branch"][branch]["t_bus"])
        from_bus = str(self.configured_network["branch"][branch]["f_bus"])

        _branch_to_data = self.config_solution["solution"]["bus"][to_bus]
        _branch_from_data = self.config_solution["solution"]["bus"][from_bus]

        branch_opt_data = pd.Series(_branch_opt_data)
        to_bus_data = pd.Series(_branch_to_data)
        from_bus_data = pd.Series(_branch_from_data)

        branch_opt_data.index = branch_opt_data.index.map(lambda k: ("branch", k))
        to_bus_data.index = to_bus_data.index.map(lambda k: ("branch_t_bus", k))
        from_bus_data.index = from_bus_data.index.map(lambda k: ("branch_f_bus", k))

        # Repeat above for branch bar.
        branch_bar = self.get_branchbar_id(branch_id=branch)

        if branch_bar in self.config_solution["solution"]["branch"].keys():
            _bar_branch_opt_data = self.config_solution["solution"]["branch"][branch_bar]
        else:
            _bar_branch_opt_data = {k: 0 for k in _branch_opt_data.keys()}

        if branch_bar not in self.configured_network["branch"].keys():
            _bar_to_opt_data = {k: 0 for k, v in _branch_to_data.items()}
            _bar_from_opt_data = {k: 0 for k, v in _branch_from_data.items()}
        else:
            bar_to_bus = str(self.configured_network["branch"][branch_bar]["t_bus"])
            bar_from_bus = str(self.configured_network["branch"][branch_bar]["f_bus"])
            _bar_to_opt_data = self.config_solution["solution"]["bus"][bar_to_bus]
            _bar_from_opt_data = self.config_solution["solution"]["bus"][bar_from_bus]

        bar_branch_opt_data = pd.Series(_bar_branch_opt_data)
        bar_to_bus_data = pd.Series(_bar_to_opt_data)
        bar_from_bus_data = pd.Series(_bar_from_opt_data)

        bar_branch_opt_data.index = bar_branch_opt_data.index.map(lambda k: ("branch_bar", k))
        bar_to_bus_data.index = bar_to_bus_data.index.map(lambda k: ("branch_bar_t_bus", k))
        bar_from_bus_data.index = bar_from_bus_data.index.map(lambda k: ("branch_bar_f_bus", k))

        if self.contains_busbars:
            return_df = pd.concat(
                [parameter_data, branch_opt_data, to_bus_data, from_bus_data, bar_branch_opt_data, bar_from_bus_data,
                 bar_to_bus_data])
        else:
            return_df = pd.concat([parameter_data, branch_opt_data, to_bus_data, from_bus_data])

        return return_df

    def get_bus_state(self, bus: str):
        pass

    def get_load_state(self, load: str):
        pass

    def get_gen_state(self, gen: str):
        pass

    def get_network_cost(self) -> float:
        return self.solution["objective"]

    def get_configured_cost(self) -> float:
        return self.config_solution["objective"]

    def solve_configuration(self) -> float:
        self.config_solution = solve_opf(self.configured_network)
        return self.config_solution["objective"]

    def reset_configuration(self) -> None:
        self.configuration.reset()
        self.config_solution = copy.deepcopy(self.solution)
        self.configured_network = copy.deepcopy(self.network)

    def solve_network_configuration(self, binary_configuration: int) -> float:
        self.apply_network_configuration(binary_configuration)
        return self.solve_configuration()

    def solve_branch_configuration(self, binary_configuration: int, branch_id: str) -> float:
        self.apply_branch_configuration(binary_configuration, branch_id)
        return self.solve_configuration()

    def apply_network_configuration(self, binary_configuration: int) -> None:

        # Update configuration.
        self.configuration.process_binary(binary_configuration)

        # Modify configured network to reflect this configuration.
        self.process_changes()

    def process_changes(self) -> None:

        for change in self.configuration.get_changes():
            self.process_change(change)

        self.configuration.clear_changes()

    def process_change(self, change: ConfigurationDelta) -> None:

        component_id = str(change.id)

        if change.type == "branch":

            if change.branch_config == "from_bus":

                old_f_bus = str(self.get_bus_id(self.configured_network["branch"][component_id]["f_bus"]))

                if change.value == 1:
                    self.configured_network["branch"][component_id]["f_bus"] = int(self.get_busbar_id(old_f_bus))
                else:
                    self.configured_network["branch"][component_id]["f_bus"] = int(old_f_bus)

                if self.contains_busbars:
                    self.duplicate_busbar_branches(branch_ids=[component_id],
                                                   branch_dicts=[self.configured_network["branch"][component_id]],
                                                   bus_id=old_f_bus)

            elif change.branch_config == "to_bus":

                old_t_bus = str(self.get_bus_id(self.configured_network["branch"][component_id]["t_bus"]))

                if change.value == 1:
                    self.configured_network["branch"][component_id]["t_bus"] = int(self.get_busbar_id(old_t_bus))
                else:
                    self.configured_network["branch"][component_id]["t_bus"] = int(old_t_bus)

                if self.contains_busbars:
                    self.duplicate_busbar_branches(branch_ids=[component_id],
                                                   branch_dicts=[],
                                                   bus_id=old_t_bus)

            elif change.branch_config == "inactive":
                self.configured_network["branch"][component_id]["br_status"] = 1 - change.value

        elif change.type == "bus":

            busbar_id = str(self.get_busbar_id(component_id))
            adjacent_branch_ids, adjacent_branch_dicts = self.get_adjacent_branches([component_id, busbar_id])

            if change.value == 1:
                self.duplicate_busbar_branches(adjacent_branch_ids, adjacent_branch_dicts, component_id)
            else:
                self.deactivate_busbar_branches(adjacent_branch_ids)

        else:
            old_bus = self.configured_network[change.type][component_id][change.type + "_bus"]
            self.configured_network[change.type][component_id][change.type + "_bus"] = self.get_busbar_id(old_bus)

    def apply_branch_configuration(self, binary_configuration: int, branch_id: str) -> None:
        # Update configuration.
        self.configuration.process_branch_binary(binary_configuration, branch_id)

        # Modify configured network to reflect this configuration.
        self.process_changes()

    def get_busbar_id(self, bus_id: str) -> int:
        return int(bus_id) + len(self.network["bus"]) // 2

    def get_bus_id(self, busbar_id: str) -> int:
        return (int(busbar_id) - 1) % (len(self.network["bus"]) // 2) + 1

    def get_branchbar_id(self, branch_id: str) -> int:
        return int(branch_id) + len(self.network["branch"])

    def duplicate_busbar_branches(self, branch_ids: list[str], branch_dicts: list[dict], bus_id: str):
        for i, branch_id in enumerate(branch_ids):
            new_branch_dict = copy.deepcopy(branch_dicts[i])
            if new_branch_dict["f_bus"] == bus_id:
                new_branch_dict["f_bus"] = str(self.get_busbar_id(bus_id))
            else:
                new_branch_dict["t_bus"] = str(self.get_busbar_id(bus_id))

            new_branch_id = str(self.get_branchbar_id(branch_id))

            self.configured_network["branch"][new_branch_id] = new_branch_dict

    def deactivate_busbar_branches(self, branch_ids: list[str]):
        for i, branch_id in enumerate(branch_ids):
            new_branch_id = str(self.get_branchbar_id(branch_id))
            self.configured_network["branch"][new_branch_id]["br_status"] = 0

    def get_adjacent_branches(self, bus_ids: list[str]) -> (list[str], list[dict]):
        keys = [k for k, v in self.network["branch"].items() if (v["f_bus"] in bus_ids or v["t_bus"] in bus_ids)]
        values = [self.network["branch"][k] for k in keys]

        return keys, values


def recursive_dict_cast(network: dict) -> dict:
    for key, value in network.items():
        if type(value) is DictValue:
            network[key] = dict(network[key])
            recursive_dict_cast(network[key])
        elif type(value) is VectorValue:
            network[key] = list(network[key])

    return dict(network)


def solve_opf(network: dict) -> dict:
    # with open('network.pickle', 'wb') as handle:
    #     pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Solve powerflow.
    result = Main.pm_solve_opf(network)


    # Results contain jlwrap type objects that encode information about opt feasibility - convert them to string.
    # for key in result:
    #     if "jlwrap" in str(type(result[key])):
    #         result[key] = str(result[key])

    return recursive_dict_cast(result)


def load_test_case(path=None):
    if path is None:
        path = 'ieee_data\\pglib_opf_case57_ieee.m'

    return Main.load_test_case(path)
