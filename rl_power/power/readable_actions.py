def action_branch_data_to_readable(action, branch_data, n_buses) -> str:

    t_bus = int(branch_data["t_bus"]) - 1
    f_bus = int(branch_data["f_bus"]) - 1

    if action == 0:
        readable_action = f"Action {action}: (b{f_bus}, b{t_bus})"
    elif action == 1:
        readable_action = f"Action {action}: (bb{f_bus%n_buses}, b{t_bus})"
    elif action == 2:
        readable_action = f"Action {action}: (b{f_bus}, bb{t_bus%n_buses})"
    elif action == 3:
        readable_action = f"Action {action}: (bb{f_bus%n_buses}, bb{t_bus%n_buses})"
    elif action == 4:
        readable_action = f"Action {action}: (b{f_bus%n_buses}, b{t_bus%n_buses}) off"
    else:
        readable_action = "unsupported action"

    return readable_action
