The *.py* files in this folder are the parameter files that can be used to load the pre-tuned parameters to re-produce the results shown in the [paper](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894).

The [train_and_test.py](../train_and_test.py) loads the required parameters to run, load and save the network and results from the [params.py](../params.py). The [params.py](../params.py) file shows an example implemetation of these parameters with description of each parameter's use in the network. This folder contains the pre-tuned parameter files for re-producing the results shown in the [paper](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894).

### Description of the parameter files 

Here is a description of what parameters were used for each result presented in the [paper](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894).

Please refer to [DLoc paper](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894) to see the figure numbers and the appropriate results that you can reproduce

- **params_fig10a.py**: Generate the DLoc results to reproduce the plot in Figure 10a
- **params_fig10b.py**: Generate the DLoc results to reproduce the plot in Figure 10b
- **params_fig11a.py**: Generate the DLoc results to reproduce the plot in Figure 11a for without copensation decoder
- **params_fig11b.py**: Generate the DLoc results to reproduce the plot in Figure 11b for without copensation decoder
- **params_fig13a_20MHz.py**: Generate the DLoc results to reproduce the plot in Figure 13a for the 20MHz Bandwidth
- **params_fig13a_40MHz.py**: Generate the DLoc results to reproduce the plot in Figure 13a for the 40MHz Bandwidth
- **params_fig13b.py**: Generate the DLoc results to reproduce the plot in Figure 13b for the Disjoint dataset
- **params_tab1_test2.py**: Generate the DLoc results to reproduce the results in Table1, where the network is trained on Env-1/3/4 and Tested on Env-2
- **params_tab1_test3.py**: Generate the DLoc results to reproduce the results in Table1, where the network is trained on Env-1/2/4 and Tested on Env-3
- **params_tab1_test4.py**: Generate the DLoc results to reproduce the results in Table1, where the network is trained on Env-1/2/3 and Tested on Env-4

### Loading the parameters

As mentioned earlier, [train_and_test.py](../train_and_test.py) file loads the required parameters from [params.py](../params.py). So to load the pre-tuned parameters to re-generate the results from the paper, please copy the correponding parameter files in this folder to params.py file in the parent folder.

```params
cp <params_file.py> ../params.py
```

Then you can run the [train_and_test.py](../train_and_test.py) script to re-produce the results from the paper.
