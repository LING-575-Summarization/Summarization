# Summarization

Multi-Document Guided Summarization

Repository for the project component of the LING 575 - Summarization class.

Student team:

- Anna Batra
- Sam Briggs
- Junyin Chen
- Hilly Steinmetz

## Conda Guide

This guide is to set up Conda on Patas/Dryas. To download and use Conda:

1. `wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh`

2. `sh Anaconda3-2022.10-Linux-x86_64.sh`

    This will modify your `.bashrc` which you can find in your user directory (`~`). If you didn't have one before, it will create one. Your modified `.bashrc` will include code similar to the following:

    ```bash
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/home2/[USER]/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home2/[USER]/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/home2/[USER]/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/home2/[USER]/anaconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    ```
    
    > Note: Files starting with `.` are often hidden. You can list those files with `ls -a`

3. `rm Anaconda3-2022.10-Linux-x86_64.sh`

4. `source ~/.bashrc`

5. `conda env create -f environment.yml` (A simple environment!)

6. Activate the environment with `conda activate summarization`. You're good to go!

    > Note: If you use VSCode, you can select a new python interpreter with cmd+p and using the command "Python: select interpreter".

    > A helpful list of commands to manage your Conda environment can be found [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)!

### Troubleshooting

If you're running into issues sourcing the environment, you will want to move the code from the `.bashrc` information to a `~/.bash_profile` file.

If you do not have a `.bash_profile` file, then use `touch ~/.bash_profile` to create one. Copy the code in your `.bashrc` file to the `.bash_profile`. Delete the code you just copied from `.bashrc` file. Finally, add the following code to your `.bashrc` file.

```bash
source ~/.bash_profile
```
