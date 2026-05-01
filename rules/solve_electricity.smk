# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

def input_network_for_solve(w):
    # Base network prepared in resources/
    return resources(f"networks/base_s_{w.clusters}_elec_{w.opts}.nc")

def input_network_for_solve_gt(w):
    # Clustered network produced by geo_temporal_cluster_network (resources/)
    return resources(f"networks/base_s_{w.clusters}_elec_{w.opts}_gt.nc")

def input_days_assignment_for_solve_gt(w):
    # Mapping CSV produced by geo_temporal_cluster_network (resources/)
    return resources(
        f"geotemporal_clustering/base_s_{w.clusters}_elec_{w.opts}/days_assignment.csv"
    )


ruleorder: expand_gt_optimized_network > solve_network
ruleorder: solve_network_gt > solve_network

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
        network=input_network_for_solve,
    output:
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc",
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


rule solve_network_gt:
    wildcard_constraints:
        opts=r".*Gt.*"
    message:
        "Solving electricity network optimization (GT) for {wildcards.clusters} clusters and {wildcards.opts} electric options"
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        geotemporal=config_provider("clustering","geotemporal"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=input_network_for_solve_gt,
        days_assignment=input_days_assignment_for_solve_gt,
    output:
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt.nc",
        config=RESULTS + "configs/config.base_s_{clusters}_elec_{opts}.yaml",
        full_timeline=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt_full_timeline.csv",
        storage_units_t_full_soc=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt_storage_units_t_full_soc.csv",
        stores_t_full_e=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt_stores_t_full_e.csv",
    log:
        solver=normpath(
            RESULTS + "logs/solve_network/base_s_{clusters}_elec_{opts}_gt_solver.log"
        ),
        memory=RESULTS + "logs/solve_network/base_s_{clusters}_elec_{opts}_gt_memory.log",
        python=RESULTS + "logs/solve_network/base_s_{clusters}_elec_{opts}_gt_python.log",
    benchmark:
        RESULTS + "benchmarks/solve_network/base_s_{clusters}_elec_{opts}_gt"
    threads: solver_threads
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        "../scripts/solve_network.py"


rule expand_gt_optimized_network:
    wildcard_constraints:
        opts=r".*Gt.*"
    message:
        "Expanding optimized GT network back to full time series for {wildcards.clusters} {wildcards.opts}"
    input:
        network_gt=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt.nc",
        days_assignment=input_days_assignment_for_solve_gt,
        full_timeline=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt_full_timeline.csv",
        storage_units_t_full_soc=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt_storage_units_t_full_soc.csv",
        stores_t_full_e=RESULTS + "networks/base_s_{clusters}_elec_{opts}_gt_stores_t_full_e.csv",
    output:
        network=RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc",
        marker=touch(RESULTS + "networks/.base_s_{clusters}_elec_{opts}_expanded_gt.done"),
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
    

