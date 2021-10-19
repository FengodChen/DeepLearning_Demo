from test.test_swin_transformer import *

(net, logger, trainer) = get_mine_net()
trainer.train(dataloader_train, dataloader_test, compare_func, 10, 5000)
trainer.test(dataloader_test, compare_func)
