from test.test_swin_transformer import *
trainer.train(dataloader_train, 10)
trainer.eval(dataloader_test, compare_func)