# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
model_configs = {
	"name": 'MyModel',
	"save_dir": './saved_models',
	"save_interval": 20,
	"num_classes": 10,
	"first_num_filters": 16,
	"resnet_size": 3, 
	"cardinality": 8,
	"base_width": 16*4,
	"widen_factor": 4,
	"output_size": 64,
}

training_configs = {
	"learning_rate": 0.05,
	"lr_scheduler_milestones" : [50, 100, 150],
	"batch_size": 128,
	"max_epoch": 200,
	"weight_decay": 5e-4	
}
### END CODE HERE

