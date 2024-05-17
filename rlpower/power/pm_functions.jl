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
    return PowerModels.parse_file(path)
end
