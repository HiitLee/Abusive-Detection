## Config Introduction ##

- Use `ConfigParser` to config parameter  
	- `from configparser import ConfigParser`  .
	- Detail see `config.py` and `config.cfg`, please.  

- Following is `config.cfg` Parameter details.

- [Data]
	- `word_Embedding ` (True or False) ------ whether to use pretrained embedding.

	- ` word_Embedding_Path ` (path)  ------ word embedding file path(`Pretrain_Embedding`).

	- `name_trainfile /name_trainfile /name_trainfile `(path)  ------ train/dev/test data path(`Data`).

	- `min_freq` (integer number) ------ The smallest Word frequency when build vocab.

	- `shuffle/epochs-shuffle`(True or False) ------ shuffle data .

- [Save]
	
- `snapshot ` (path) ------ snapshot .
	
- `predict  ` (path) ------ predict  .
	
- `test ` (True or False) ------ test .
	
- `save_dir   ` (path) ------ save model and config file.
	
- `rm_model` (True or False) ------ remove model to save space.
	
- [Model]

	- `static ` (True or False) ------ whether to update the weight during train.
- `wide_conv  ` (True or False) ------ whether to use wide convcolution True : wide False : narrow.
	- `embed_dim ` (integer) ------ embedding dim = pre-trained embedding dim.
	- `embed-finetune` (True or False) ------ word embedding finetune or no-finetune.
	- `lstm_hidden_dim` (integer) ------number of lstm hidden.
- `lstm_num_layers ` (integer) ------number of lstm layer.
	- `word_hidden` (integer) ------number of word-lstm hidden.
- `sent_hidden ` (integer) ------number of sentence-lstm hidden.
	- `batch_normalizations ` (True or False) ------ batch_normalizations .
- `bath_norm_momentum  `(float) ------ batch_normalizations value.
	- `batch_norm_affine  ` (True or False) ------  batch_normalizations option .
- `dropout-emb/dropout `(float) ------ dropout for prevent overfitting.
	- `clip_max_norm  `(float) ------  for prevent overfitting.
- `kernel_num ` (integer) ------ cnn kernel numbers.
	- `kernel_sizes` ------ cnn kernel sizes.
- `cnn_filter_size` (set) ------ cnn filter numbers.
	- `init_weight  ` (True or False) ------  initial neural networks weight .
- `init_weight_value`(float) ------  initial neural networks weight value.
	- `cnn_kernel` (set) ------ cnn kernel numbers.
- `word_kernel` (set) ------ word-cnn kernel numbers.
	- `sent_kernel` (set) ------ sent-cnn kernel numbers.
	
	- `clstm_kernel` (set) ------ clstm1-cnn kernel numbers.
- `clstm_add_kernel` (set) ------ clstm2-cnn kernel numbers.
	- `clstm_filter_size` (integer) ------ clstn filter numbers.
	- `clstm_hidden_size` (integer) ------ number of c-lstm hidden.
	- `class_num` (integer) ------ class number.
	
- [Optimizer]

	- `lr ` (float) ------ `learning rate`
- `Adam ` (True or False) ------ `torch.optim.Adam`
	- `SGD ` (True or False)  ------ `torch.optim.SGD`
	- `Adadelta  ` (True or False)  ------ `torch.optim.Adadelta `
	- `learning_rate `(float) ------ learning rate(0.001, 0.01).
- `weight-decay` (float) ------ L2.


- [Train]

	- `num-threads` (Integer) ------ threads.
- `device ` (Integer) ------ gpu device .
	- `cuda` (True or False) ------ support `cuda` speed up.
- `epochs` (Integer) ------ train epochs
	- `batch-size` (Integer) ------ number of batch
- `log-interval`(Integer) ------ steps of print log.
	- `test-interval`(Integer) ------ eval dev/test.
- `save-interval`(Integer) ------ save model.
	- `patience`(Integer) ------ save model.