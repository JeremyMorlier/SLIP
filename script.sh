#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=log/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --partition=prepost
#SBATCH --nodes=1 # reserver 1 nœud

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate ../venvs/venCLIPRocket
set -x # activer l’echo des commandes
export WANDB_DIR=$WORK/wandb/
srun python3 make_dataset.py --output $SCRATCH/YFCC100M/ --metadata $DSDIR/YFCC100M/metadata/yfcc100m_dataset --images_ids $SCRATCH/YFCC100M/flickr_unique_ids.npy --yfcc15m_tsv $SCRATCH/YFCC100M/yfcc100m_subset_data.tsv