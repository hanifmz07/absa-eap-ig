# Dataset README

## Dataset for Training and Testing
The dataset used for training and testing are in the `eng`, `indo`, and `sunda` folder.
### Datasets version
All data below are already augmented.
- `clean`: Clean data with split 2700, 900, 900
- `clean_train2500`: Clean data with split 2500, 1000, 1000
- `clean_traintruncated`: Clean data with split 2482 1000 1000, all instance (input + target) with length more than 300 tokens are dropped
- `corrected_all_alts`: Corrected data (based on the data correction guideline) from `clean_traintruncated` with all alternative solution inserted

## Counterfactual Dataset
`empty_counterfacts` folder are used to store selected test data used for counterfactual data generation.

`counterfacts` folder are used to store valid counterfactual data that are ready to be used for circuit discovery.
- `counterfactsv1`: A and O must be related, A can be other aspect but still related to the original A by category, O must have the opposite sentiment compared to the original O
- `counterfactsv2`: A and O must be unrelated one another, A must be an aspect that is unrelated by category with the original A, O must have the opposite sentiment compared to the original O
- `counterfactsv2.1`: same as v2, but includes instances that have 2 or 3 triplets.