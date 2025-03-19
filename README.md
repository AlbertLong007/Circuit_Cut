# Measure-and-Prepare Based Quantum Circuit Cut Algorithm

The code by Team **QA** in [3rd CCF Sinan Cup](https://learn.originqc.com.cn/contest), won **Outstanding Award** in quantum-supercomputing track (**National Only**).

## Dependencies

You'll need a working Python environment to run the code. `numpy`, `pyqpanda` and `pyvqnet` modules are needed. All the experiments we have done are run in the `conda` virtual environment with Python 3.9.

To build and test the program, produce all results and figures, run this in the top level of the repository:

    pip install -r requirements.txt

If all goes well, all dependencies will be installed successfully.

## Examples of how to use the code

The model we built saves in file `cutcircuit.py`. To test the model, run 

    python eval.py

The program will load trained parameters saved in file `weights.pt`. On given test dataset, the final accuracy will be printed out in shell.

To train the model, run the code

    python train.py

The parameters trained will save in file `weights.pt`.
