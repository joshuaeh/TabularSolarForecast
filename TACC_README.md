# TACC Guide

Within this guide, `<input description>` will be used to show values you will need to put yourself. For example, my Github username is joshueh, so if I saw the prompt `ssh <github username>@github.com`, I would write `ssh joshuaeh@github.com`.  

Currently, the project has the following allocations:

- Maverick2: 1,000 node hours  
  - 24 nodes with 4 NVIDIA 1080 Ti GPUs per node.  
  - 4 nodes with 2 NVIDIA v100 GPUs which are theoretically twice as powerful as the 1080 TI
- Lonestar6: 100 node hours  
  - 16 of the 560 noces have dual NVIDIA a100 GPUs which are theoretically twice as powerful as the v100
- Longhorn: 100 node hours
  -96 nodes with 4 v100 GPUs  

It should be fairly easy to get more.

## Linux Commands to know

You will generally use the command line to interface with TACC. On Linux of MAcOS this is easy with the default bash or zsh terminal. On windows, you will need to find ways to use bash and other scripts.

### Bash

You will interact with the computers on TACC using bash: the Bourne-Again SHell and Linux CLI. Don't worry about knowing everything, but know how to figure things out. A noncomprehensive list of useful commands includes:  

- `cd`: short for change directory and will change the active directory
- `pwd`: will print the working directory
- `ls`: will list the contents of a directory, the `-a` flag will include hidden files or directories
- `cp`: will copy a file in the format `cp <source> <destination>`. The `-R` flag makes the script recursive for directories

### tar

tar is a command to compress a group of files into an archive.  

-`-c` is to compress. syntax: `tar -c <result filename> <path of object to be compressed>`  
-`-x` expand  
-`-v` Verbose  
-`-z` .gz flag  
-`-f` File  
If the path of the object to be compressed is not in the cwd, use the `-C` flag.

#### Examples

compress the directory `project/` into a tar.gz archive named `helloArchive.tar.gz` in the directory `/temp/`:  
`tar -zcvf /temp/helloArchive.tar.gz project/`

## Work Environments

When you log in, you will be on a login node. Do not use this node for computation, this is the fastest way to get your account suspended. Instead, use these nodes to manage files and compile. Use compute nodes for computations.  

There are 4 main filesystems on TACC depending on whcih system you are on:  
| File system | Best activities                                 | Example usage                                                                                 | File limit                                   | File persistance                                                   | Available on       |
|-------------|-------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------|--------------------------------------------------------------------|--------------------|
| $HOME       | NOT FOR HEAVY I/O OPERATIONS Compiling, editing | Cron jobs, small scripts, environment settings                                                | 10 GB, 200,000 files across all TACC systems | Permanent                                                          | All                |
| $WORK       | NOT FOR HEAVY I/O OPERATIONS Staging datasets   | software installations, original datasets that can't be reproduced, job scripts and templates | 1TB, 3,000,000 files across all TACC systems | Permanent                                                          | All                |
| $SCRATCH    | data manipulation, I/O, working                 |                                                                                               | Effectively none                             | Liable to be deleted if not accessed in 10 days but rarely happens | The current system |
| /tmp/       | data manipulation, I/O, working                 |                                                                                               | Depends on system                                  | Deleted at the end of each compute session                         |                    |

## Accessing TACC

Before you are able to access a login node on TACC resources you will need:  

1. To be on an active allocation
2. Have Multi-Factor Authentication enabled
3. Somewhere to ssh

The command to ssh into TACC will look like: `ssh <TACC username>@maverick2.tacc.utexas.edu` for the maverick2 system. After a successful promt, you will be prompted for your password, and finally the 2FA code. Note that as a linux system, it will not show you a character for each character you type, but instead it will remain blank.

| TACC system | urls                      |
|-------------|---------------------------|
| Maverick2   | maverick2.tacc.utexas.edu |
| lonestar6   | ls6.tacc.utexas.edu       |
| longhorn    | longhorn.tacc.utexas.edu  |

To transfer a file from your computer to TACC, use the `scp` command: `scp <path to file to be sent> <username>@<tacc system>.tacc.utexas.edu:<path to destination>`

## Configuring your project

1. Configure your local git instance on a login node using `git --config user.name <your user name>` and `git --config user.email <your github email>`. If you need to use ssh keys to access your github account, you can do so using the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)  
   1.1 __NOTE:__ you should not run `ssh-keygen` if you do not have your ssh keys in the `.ssh/` directory. Instead, run `mv .ssh dot.ssh.old`, log out, and log back in.  
2. Clone your project into your WORK file system

## Connecting to a compute node

There are two ways to connect to a compute node: using `idev -m <minutes>` for an interactive CLI session, or by submitting a SLURM script. DO NOT do computation on the login nodes, or we will get in big trouble. Also limit the I/O and in principle, only read/write from `$WORK` or `$HOME` at the beginning or end of a job.

## Transfering files to and from a compute node

Transfering one archive will go much faster than transferring a bunch of files. To do this, compress what you will transfer into a tarball, then move to either `/tmp` or `$SCRATCH` depending on the system you are working on.

## Scripting jobs with SLURM

The slurm script instructs the queue what to do. Here is an annotated example script and the accompanying output:

```shell
#
# https://portal.tacc.utexas.edu/user-guides/maverick2
#--------------------------------------------------------------------------
# ----------------------------- General Notes ----------------------------
#--------------------------------------------------------------------------
#
# In general, this is a bash script with specific comments starting with 
# "#SBATCH" interpreted as commands to the TACC queue system. Some general
# notes: "#" is a comment. The SBATCH -o command specifies the output file
# it cannot have spaces in its name. Path is relative to the slurm text file
# that is submitted, or rather, the working directory when 
#`sbatch <slurm text file>` is called.
#
# Other useful tips: use `squeue` to show the queue. Since you cannot modify
# the job once submitted, use `cancel <job id>` to cancel a job in the queue.
# To follow job progress, if my output file were "example.out", I could print
# the last few lines to the terminal and follow progress using the command 
# `tail -f example.out`
#
#--------------------------------------------------------------------------
# --------------------------- commands to sbatch --------------------------
#--------------------------------------------------------------------------
#SBATCH -J StellarDL                            # Job name
#SBATCH -o out/slurmtest-%j.out                 # Name of stdout output file (%j expands to jobId)
#SBATCH --mail-user=joshua.hammond@utexas.edu   # email address for communications
#SBATCH --mail-type=ALL                         # Options include NONE, ALL, END, FAIL
#SBATCH -p gtx                                  # Queue name
#SBATCH -N 1                                    # Total number of nodes requested (68 cores/node)
#SBATCH -n 1                                    # Total number of mpi tasks requested
#SBATCH -t 13:00:00                             # Run time (hh:mm:ss)

#--------------------------------------------------------------------------
# --------------------------- bash script below --------------------------
#--------------------------------------------------------------------------
#
# last update: joshuaeh 20220706

# info for debugging
echo "============================================"
echo "TACC: job $SLURM_JOB_ID execution at: `date`"

# add date, host, directory for later troubleshooting
date; hostname; pwd

# our node name
NODE_HOSTNAME=`hostname -s`
echo "TACC: running on node $NODE_HOSTNAME"
echo "============================================"

# create tarball and transfer to /tmp/
cd $WORK

echo "create tarball"
tar -zcf /tmp/stellar.tar.gz stellar/
echo "done"

cd /tmp/

# extract tarball in /tmp/ 
echo "extract tarball"
tar -xzf stellar.tar.gz
echo "done"

cd stellar/

# create python virtual environment
echo "Create python environment"
python3 -m venv --copies .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "done"

cd data_preprocessing/

# download moving MNIST
echo "moving MNIST"
python download_Moving_MNIST_data.py
echo "done"

# download NREL Weather
echo "NREL Weather"
python download_weather_data_NREL.py
echo "done"

# download NREL ASI
echo "NREL ASI"
python download_ASI_data_NREL.py
echo "done"

# remove virtual environment (about 3GB of data)
rm -R /tmp/stellar/.venv/

# create tarball on work drive
echo "create tarball"
cd /tmp/
tar -zcf $WORK/stellar.tar.gz  stellar/
echo "done"

echo "complete"

echo "============================================"
echo "TACC: job $SLURM_JOB_ID execution finished at: `date`"
```

This is submitted using `sbatch <path to script>`

## Previous Errors and Solutions

### Problem: Minimize I/O with python modules installed on $HOME

Solution: create a virtual environment and using the `--copies` argument to create copies of the modules within the local environment

### Problem: A package will not upgrade to the level needed

Solution: Upgrade pip using `pip install --upgrade pip`. Also see the next problem.

### Problem: the wheels for installing a package cannot compile

Two things here: First, make sure to upgrate wheels and setup tools when upgrding pip: `pip install --upgrade pip setuptools wheel`. Next, make sure that the python packages from other modules are not conflicting by using module unload... To get the STELLAR scripts to work I had to unload python2, create the virtual environment, load python3 so that I had a cmake compiler, pip install requirements, then unload python3 and pip install h5py as otherwise it would not upgrade h5py as it was seen within the python3 module.  

### Problem: problems creating tar.gz

Check the order of the flags given. Order matters.
