import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net', required=True, type=str, help='the name of the net')
parser.add_argument('--ref', action='store_true', help='using refer net instead of my net')
parser.add_argument('--plot-train', action='store_true', help='plot train log')
parser.add_argument('--plot-eval', action='store_true', help='plot eval log')
parser.add_argument('--dry-run', action='store_true', help='dry run')
parser.add_argument('--eval-epoch-num', type=int, default=10, help='eval epoch')
parser.add_argument('--epoch-num', type=int, default=100, help='epoch num')

opt = parser.parse_args()

if opt.net == "swin_transformer":
    from test.test_swin_transformer import *
    if opt.ref:
        (net, logger, trainer) = get_ref_net()
        trainer.train(dataloader_train, dataloader_test, compare_func, opt.eval_epoch_num, opt.epoch_num)
        trainer.test(dataloader_test, compare_func)
    else:
        (net, logger, trainer) = get_mine_net()

    if opt.plot_train:
        logger.plot_log("train", "show")
    elif opt.plot_eval:
        logger.plot_log("eval", "show")

    if not opt.dry_run:
        trainer.train(dataloader_train, dataloader_test, compare_func, opt.eval_epoch_num, opt.epoch_num)
        trainer.test(dataloader_test, compare_func)

