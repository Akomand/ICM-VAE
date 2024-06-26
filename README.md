# Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms
This is the source code for the implementation of "Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms" (IJCAI 2024)

Learning disentangled causal representations is a challenging problem that has gained significant attention recently due to its implications for extracting meaningful information for downstream tasks. In this work, we define a new notion of causal disentanglement from the perspective of independent causal mechanisms. We propose ICM-VAE, a framework for learning causally disentangled representations supervised by causally related observed labels. We model causal mechanisms using learnable flow-based diffeomorphic functions to map noise variables to latent causal variables. Further, to promote the disentanglement of causal factors, we propose a causal disentanglement prior that utilizes the known causal structure to encourage learning a causally factorized distribution in the latent space. Under relatively mild conditions, we provide theoretical results showing the identifiability of causal factors and mechanisms up to permutation and elementwise reparameterization. We empirically demonstrate that our framework induces highly disentangled causal factors, improves interventional robustness, and is compatible with counterfactual generation.



## Usage

### Training and evaluating 

1. Clone the repository

     ```
     git clone https://github.com/Akomand/ICM-VAE.git
     cd ICM-VAE
     ```

2. Create New Environment

    ```
    conda env create -f environment.yml
    ```

3. Activate environment

    ```
    conda activate icm_vae
    ```

4. Generate data and place in data/ subdirectory


5. Navigate to `experiments` folder

    ```
    cd experiments
    ```

6. Run training script

    ```
    python train_[dataset].py
    ```


### Data acknowledgements
Experiments are run on the following datasets to evaluate our model:

#### Datasets
<details closed>
<summary>Pendulum Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>Flow Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>CausalCircuit Dataset</summary>

[Link to dataset](https://developer.qualcomm.com/software/ai-datasets/causalcircuit)
</details>

## Citation

If you use our code or think our work is relevant to yours, we encourage you to cite this paper:

```bibtex
@inproceedings{
komanduri2024learning,
title={Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms},
author={Aneesh Komanduri and Yongkai Wu and Feng Chen and Xintao Wu},
booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
year={2024}
}
```


## Acknowledgement
This work is supported in part by National Science Foundation under awards 1910284, 1946391
and 2147375, the National Institute of General Medical Sciences of National Institutes of Health
under award P20GM139768, and the Arkansas Integrative Metabolic Research Center at University
of Arkansas.
