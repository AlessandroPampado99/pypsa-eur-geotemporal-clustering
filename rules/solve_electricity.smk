# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

def input_network_for_solve(w):
    # Hardcoded: geotemporal reduction is activated when "Gt" appears in opts.
    if "Gt" in str(w.opts):
        return resources(f"networks/base_s_{w.clusters}_elec_{w.opts}_gt.nc")
    return resources(f"networks/base_s_{w.clusters}_elec_{w.opts}.nc")

def output_network_for_solve(w):
    if "Gt" in str(w.opts):
        return RESULTS + f"networks/base_s_{w.clusters}_elec_{w.opts}_gt.nc"
    return RESULTS + f"networks/base_s_{w.clusters}_elec_{w.opts}.nc"

def days_assignment_for_solve(w):
    """
    For GT runs return the real mapping file. For non-GT return a dummy path
    (won't be used because seasonal storage extra_functionality is only enabled for GT).
    """
    if "Gt" in str(w.opts):
        return RESULTS + f"geotemporal_clustering/base_s_{w.clusters}_elec_{w.opts}/days_assignment.csv"
    return RESULTS + "geotemporal_clustering/_dummy_days_assignment.csv"

rule solve_network:
    message:
        "Solving electricity network optimization for {wildcards.clusters} clusters and {wildcards.opts} electric options"
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        # If GT: solve the clustered network produced by geo_temporal_cluster_network.
        # Else: solve the standard network.
        network=lambda w: (
            RESULTS + f"networks/base_s_{w.clusters}_elec_{w.opts}_gt.nc"
            if "Gt" in str(w.opts)
            else input_network_for_solve(w)
        ),
        # Needed only for GT seasonal storage, but always provided (dummy for non-GT).
        days_assignment=days_assignment_for_solve,
    output:
        # IMPORTANT: for GT, output must remain *_gt.nc (optimized clustered net)
        network=output_network_for_solve,
        config=RESULTS + "configs/config.base_s_{clusters}_elec_{opts}.yaml",
    log:
        solver=normpath(
            RESULTS + "logs/solve_network/base_s_{clusters}_elec_{opts}_solver.log"
        ),
        memory=RESULTS + "logs/solve_network/base_s_{clusters}_elec_{opts}_memory.log",
        python=RESULTS + "logs/solve_network/base_s_{clusters}_elec_{opts}_python.log",
    benchmark:
        (RESULTS + "benchmarks/solve_network/base_s_{clusters}_elec_{opts}")
    threads: solver_threads
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        "../scripts/solve_network.py"

rule expand_gt_optimized_network:
    message:
        "Expanding optimized GT network back to full time series for {wildcards.clusters} {wildcards.opts}"
    input:
        # This will be *_gt.nc only when opts contains 'Gt'
        network_gt=output_network_for_solve,
        # Real CSV for GT, dummy for non-GT (but rule will only run if needed)
        days_assignment=days_assignment_for_solve,
    output:
        # Always the standard network name (used by postprocess rules)
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc"
    log:
        RESULTS + "logs/geo_temporal_clustering/expand_gt_{clusters}_{opts}.log"
    benchmark:
        RESULTS + "benchmarks/geo_temporal_clustering/expand_gt_{clusters}_{opts}"
    threads: 1
    resources:
        mem_mb=4000
    script:
        "../scripts/geo_temporal_clustering/expand_gt_network.py"

rule solve_operations_network:
    message:
        "Solving electricity network operations optimization for {wildcards.clusters} clusters and {wildcards.opts} electric options"
    params:
        options=config_provider("solving", "options"),
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc",
    output:
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}_op.nc",
    log:
        solver=normpath(
            RESULTS
            + "logs/solve_operations_network/base_s_{clusters}_elec_{opts}_op_solver.log"
        ),
        python=RESULTS
        + "logs/solve_operations_network/base_s_{clusters}_elec_{opts}_op_python.log",
    benchmark:
        (RESULTS + "benchmarks/solve_operations_network/base_s_{clusters}_elec_{opts}")
    threads: 4
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        "../scripts/solve_operations_network.py"
    

