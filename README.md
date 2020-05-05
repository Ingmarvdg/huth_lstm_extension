

This notebook is made for the course Advanced fMRI Analysis at Sungkyunkwan University, it explains how to make an alternative context representation model that can be used to do the experiment done by [A. Huth](https://papers.nips.cc/paper/7897-incorporating-context-into-language-encoding-models-for-fmri). 

To use the LSTM for word representations instead of the Eng1000 representation, you will have to make the following adjustments to the original tutorial:

1. load the glove word embeddings (cell 3).
2. load the pretrained models (cell 10 and 11).
3. copy the new semantic model function (cell 13).
4. use the lstm semantic model function instead of the original function.

If you dont have any pretrained models, you can run the whole notebook once, it will train the models for you, but might take a while. The files and directories you will need are:

1. a models folder with the pretrained models.
2. the word embeddings .txt file.
3. the nlp_utils.py file.

I might improve on this explanation by uploading the modified tutorial, or a more detailed walkthrough. But for now, this will have to do.
