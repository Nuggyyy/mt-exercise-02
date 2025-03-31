
# Graph Analysis

##  Connections between the training, validation and test perplexity
0. In general both graphs have a similar feel to them
1. Training Perplexity
	- No dropout achieves the lowest training PPL: This model learns the data very well
	- Higher dropout rates result in higher training PPL but the dropout rate help reduce overfitting
2. Validation Perplexity
	- Dropout 0.25 and 0.5 yield the lowest validation PPL. This means that these settings generalize better to unseen data
	- Dropout of 0 performs similarly to 0.25 but both are slightly better than 0.5. If there were more epochs, we guess that 0.25 and 0.5 will end up performing better than 0.
	- High dropout rates have high validation ppl which could mean underfitting
3. Connections
	- Dropout 0.25 and 0.5 seem to get lower quicker than the rest which could indicate that the models can generalize better than others
	- Models with high dropout (e.g. 1.0) struggle to lower their PPL which indicates that the model struggles to learn effectively
	- Models with very low dropout (e.g. 0) have low PPL in training and in validation. This shows that the model appearently generalizes well but in our case this might be because of overfitting
	
## Best dropout setting
In our case the best dropout setting appears to be 0.25 or 0.5.

# Sample Analysis
