CONFIG_DIR := config/final_configs_optimization
SNAKEMAKE := snakemake -j64 all_electric --rerun-incomplete

.PHONY: all IT IT_200 IT_900 IT_1600 DE ES

all: IT IT_200 IT_900 IT_1600 DE ES

IT:
	$(SNAKEMAKE) --configfile $(CONFIG_DIR)/config_IT.yaml

IT_200:
	$(SNAKEMAKE) --configfile $(CONFIG_DIR)/config_IT_200.yaml

IT_900:
	$(SNAKEMAKE) --configfile $(CONFIG_DIR)/config_IT_900.yaml

IT_1600:
	$(SNAKEMAKE) --configfile $(CONFIG_DIR)/config_IT_1600.yaml

DE:
	$(SNAKEMAKE) --configfile $(CONFIG_DIR)/config_DE.yaml

ES:
	$(SNAKEMAKE) --configfile $(CONFIG_DIR)/config_ES.yaml
