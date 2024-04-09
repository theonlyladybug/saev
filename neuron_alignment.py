"""
load the model
extract the MLP out weights for the specific layer
extract the enocder matrix for the SAE
normalise along the associated dimensions
multiply the two together - check whether things need transposing
take the absolute value of the result
max over the MLP dimension - should be left with a vector.
plot the resulting values in a histogram plot in plotly. between 0 and 1
"""

print('hello world')