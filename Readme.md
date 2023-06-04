# GSRSL: Gated Session-based Recommender integrating Short and Long-term preferences 

This is the implementation of our paper: <span style="color:blue">Integration of short and long-term interests: A preference aware session-based recommender</span> using **pytorch**

Briefly, the method uses items, along with their categories and features like price, to predict the next item. Additionally, a dynamic way of selecting short-term interests shows better performance compared to the baseline models.

Here is the link for the diginetica dataset used in our paper. 
- DIGINETICA: <https://competitions.codalab.org/competitions/11161#learn_the_details-data2>
   
After downloading the datasets, you can extract the files in the folder `datasets/`:



## Usage

You need to run the file  `data/preprocess.py` first to preprocess the data. 

Then you can run the file `main.py` to train the model. The model hyperparameters can be changed from the  `main.py` file itself.

## Requirements

- Python version > 3.8
- PyTorch > 1.10
- numpy > 1.20
- pandas > 1.3
- tensorflow == 2.7.0
- math

## Citation

Please cite our paper if you use the code.

