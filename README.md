# Forked Code

The original code was forked from Rafael Patronilo's thesis code repository (https://github.com/rafael-patronilo/thesis-code/tree/main). Thus
all credit goes to him for allowing other people to make use of it(myself included),
providing a baseline of scripts that already were proven to work and can be continued upon.
With that said, i tried to match the original workflow as best as possible, however
I know that navigation the sea of scripts and code might be difficult without proper 
documentation, which is the case.


# How to run

1. Install docker
2. Clone the repository
3. Run the `./docker-build` bash script
4. Run `./docker-run IMG python src/__init__.py --help`.

Docker scripts are written in bash and expect a unix environment 
(they were run in a WSL2 Ubuntu environment). Both scripts
handle the removal of dangling images/containers.

Previously, the docker-run script mounts the appropriate directories, however i changed it to use local directories. 
The script will attempt to reserve all available graphics cards for the docker container. 
This may lead to issue if you have multiple graphics cards and a few working on other tasks. 
There is currently not an option to configure this but it should be easy to edit the script.

To reproduce the experiments regarding the GTSRB-Concepts Dataset, you will need to obtain the [GTSRB-Concepts dataset]. However, this is not enough
and so to ease the process of building all the correct files, with all the necessary columns and several tweaks made, i made available the data folder used in
().

# Repository structure

Directory `src/core` contains tools are agnostic to the dataset used.
Remaining code is mostly specific to the XTRAINS dataset (untouched from fork) and GTSRB dataset. 

Directory `src/scripts/datasets/gtsrb` contains scripts regading the dataset in itself. A lot of the remaing script present are deprecated, in the sense that they were used once to improve the initial dataset. However there two important ones:
- `gen_gtsrb_ontology_dataset_NewOptimization.py` is the script to generate all the valid samples via the ontology rules with a simple binary generator that has to follow some more rules detailed in the dissertation, namely the symbol ones.
- `visualize_concept_noise.py` is the script to visualize in practice the concept noise in the Artificial Random Reasoning Dataset. Example of usage: - `./docker-run IMG python src/__init__.py   --models-path storage/studies datasets gtsrb visualize_concept_noise`

Directory `src/scripts/eval/gtsrb` contains scripts to produce justifications(Concept Mapping), and the concept grounding.
- `justifications.py` is the script to produce justification for Concept Mapping. Example of usage: `./docker-run IMG python src/__init__.py   --models-path storage/studies   eval gtsrb justifications   Best_Models_Family/IntermediateLoss/C2_L128_ONTO_BASE_SKIP_extremes_preRN_run0`. To use attributions, they are in this type: --attribution 'CircularShape,SymbolD1a5,SymbolVerticalLightSignals' and so on.
- `pn_concept_correspondence.py` is the script to create the basis of concept grounding. Example of usage: `./docker-run IMG python src/__init__.py   --models-path storage/studies eval gtsrb pn_concept_correspondence Best_Models_Family/IntermediateLoss/C2_L128_ONTO_BASE_SKIP_extremes_preRN_run0`
- `summarize_concept_correspondence.py`  is the script for architecture-imposed and greedy atributions in concept grounding. Example of usage: `./docker-run IMG python src/__init__.py   --models-path storage/studies eval gtsrb summarize_concept_correspondences storage/studies/Best_Models_Family/C2_L128_ONTO_BASE_untRN_run4/results/concept_cross_metrics balanced_accuracy.csv`
- `random_concept_correspondence.py` is the script for the random one-to-one concept grounding attribution.

Directory `src/scripts/gtsrb/Categories` contains the scripts to run all the networks described in the dissertation.

Directory `storage` contains files relative to each experiment. More relevant:
- `storage/studies/Best_Models_Family` contains the models and detailed results for each Hybrid Network that achieved best results regarding a type of family, whose results are summarized in the dissertation
- `storage/studies/gtsrb_rn...` contains several reasoning networks that suffered pretraining that would later be used in the Hybrid Networks, whose results are summarized in the dissertation.
- `storage/studies/gtsrb_hn...` contains several Hybrid Netwoks that were used in the following experiments in the dissertation, whose results are summarized in it.

Directory `Clingo` contains files relative to the Clingo segment of verifying the ontology valid samples. More relevant:
- `clingoCommands.txt` contains the three commands to properly do the experiment.
- `prove_valid_ontology_rows_gtsrb.lp` contains the clingo code of the verification.
- `.py` files contain the necessary scripts to easy visualization of the desired match between the generator and clingo segments.


