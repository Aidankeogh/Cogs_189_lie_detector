# Cogs_189_lie_detector

There are a couple parts here. 

In seq GRU we have code for a GRU sequence classifier. Obviously it works with the EEG data but we also tried it out with things like text classification. 

In the JSON folder we have the EEG data from all of our participants stored as json files. 

In data loader we have code to read in those json files into python dictionaries. 

In main, we have basic code for z-scoring the data, and then we call the functions from seq GRU. 
