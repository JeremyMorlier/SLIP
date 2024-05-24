#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=log/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=100:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=sxq@v100 # comptabilite V100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load anaconda-py3/2023.09
conda activate ../venvs/venCLIPRocket
set -x # activer l’echo des commandes
export WANDB_DIR=$WORK/wandb/
srun python3 make_dataset.py --output $SCRATCH/YFCC100M/ --metadata $DSDIR/YFCC100M/metadata/yfcc100m_dataset --images_ids $SCRATCH/YFCC100M/flickr_unique_ids.npy --yfcc15m_tsv $SCRATCH/YFCC100M/yfcc100m_subset_data.tsv