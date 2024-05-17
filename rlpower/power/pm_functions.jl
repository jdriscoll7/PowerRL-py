using PowerModels
# using PowerPlots
using Ipopt
using JuMP
using Printf
using JSON
# using Plots
# using VegaLite


function _convert_to_pm_dict_type(network)

    network = Dict{String, Any}(network);

    for key in keys(network)
        if network[key] isa Dict
            network[key] = Dict{String, Any}(network[key]);
            for sub_key in keys(network[key])
                if network[key][sub_key] isa Dict
                    network[key][sub_key] = Dict{String, Any}(network[key][sub_key]);
                end
            end
        end
    end
    return network;
end


function pm_solve_opf(network::Dict)
    network = _convert_to_pm_dict_type(network);
    return solve_opf(network, ACPPowerModel, JuMP.optimizer_with_attributes(Ipopt.Optimizer, "max_iter"=>150, "print_level"=>0), setting = Dict("output" => Dict("duals" => true)));
end


function load_test_case(path)

    network = PowerModels.parse_file(path);
    return network;

end


function make_busbar_network(network)

    n_buses = length(network["bus"]);

    for bus_i in 1:n_buses
        new_bus_number = bus_i + n_buses;
        new_bus_id = string(new_bus_number);
        if !haskey(network["bus"], new_bus_id)
            network["bus"][new_bus_id] = deepcopy(network["bus"][string(bus_i)]);
            network["bus"][new_bus_id]["source_id"][2] = new_bus_number;
            network["bus"][new_bus_id]["index"] = new_bus_number;
            network["bus"][new_bus_id]["bus_i"] = new_bus_number;
            bus_type = network["bus"][string(bus_i)]["bus_type"];
            if bus_type < 3
                 network["bus"][new_bus_id]["bus_type"] = network["bus"][string(bus_i)]["bus_type"];
            else
                if bus_i in [y["gen_bus"] for (x, y) in network["gen"]]
                    network["bus"][new_bus_id]["bus_type"] = 2;
                else
                    network["bus"][new_bus_id]["bus_type"] = 1;
                end
            end
        end
    end

    return network;

end
