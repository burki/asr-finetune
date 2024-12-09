# Installation 

1. Crate a folder of your choice in you $HOME HPC directory.
2. Pull the repo into the folder.
3. Install packages in the [requirements.txt](requirements.txt).
   Option 1: Conda. `conda create --name <env_name> --file requirements.txt`
   Option 2: Create a python environment and install with pip: `pip install -r requirements.txt within the environment`
4. Add the data you want to use in the folder [data](data) in your home directory. 

# Submit a job

0. Activate your environment in your preferred way. E.g. in the `.bash_profile`, within the `.sh` script, or in terminal
1. Create a config file in [configs](finetune/configs). Easiest way: copy and paste an existing `.config` file and 
   adjust some settings, e.g. [train_whisper_tiny_BOHB.config](finetune/configs/train_whisper_tiny_BOHB.config)
2. Create `.sh` script in root folder to submit a job.
3. Submit a job with sbatch, e.g. `sbatch fine_tine_tiny_BOHB.sh`

*Some notes*: 
- All relevant files are automatically saved in the scratch folder [/scratch/USERNAME/](/scratch/USERNAME/). Results of the 
submitted job with defined `output_tag` are stored in [/scratch/USERNAME/ray_results/output_tag](/scratch/USERNAME/ray_results/output_tag) and the temporary
files are automaticall stored in [/scratch/USERNAME/tmp](/scratch/USERNAME/tmp) 
- For runs on you local machine for debugging, see the next section.

# Debugging
For debugging, activate the debug flag in the config. In that case, the data from the `path_to_data` argument is loaded.
I recommend creating a debug data-folder with a small dataset to make debugging more efficient. 

To run the code on your local machine, store run_on_local_machine = True, see [tiny_debug.config](finetune/configs/tiny_debug.config)
for an example. You need to specify also a storage path under `args.storage_path`.

# Track progress in tensorboard

To track the progress of your experiment, log into you HPC account forwarding port 6007 onto you local machine through
`ssh -L 16006:127.0.0.1:6007 USER@curta.zedat.fu-berlin.de`

Run `tensorboard --logdir /scratch/USER/ray_results/output_tag/` where output_tag is again the one from the config file.

# Useful formulas

Here are some formulas to understand how many training steps are needed and how many iterations are needed (relevant 
for undertanding the tensorboard loggings)

total_Gradient_steps = round_up(length_train_set / per_device_train_batch_size) * num_epochs

iterations = round_up(total_Gradient_steps / save_steps)
