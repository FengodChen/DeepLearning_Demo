from test.test_swin_transformer import *

(net, logger, trainer) = get_mine_net()
trainer.train(dataloader_train, 10)
trainer.eval(dataloader_test, compare_func)
