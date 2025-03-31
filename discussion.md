# Graph Analysis

##  Connections between the training, validation and test perplexity
1. Training Perplexity
	- No dropout achieves the lowest training PPL: This model learns the data very well
	- Higher dropout rates result in higher training PPL but the dropout rate help reduce overfitting
2. Validation Perplexity
	- Dropout 0.25 and 0.5 yield the lowest validation PPL. This means that these settings generalize better to unseen data
	- Dropout of 0 performs similarly to 0.25 but slightly better than 0.5
	- High dropout rates have high validation ppl which could mean underfitting
3. Connections
	

## Best dropout setting
