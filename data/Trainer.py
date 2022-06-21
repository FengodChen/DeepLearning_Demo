from tqdm import tqdm

class Trainer():
    def __init__(self, net, loss, opt, dev, logger, checkpoint_epoch):
        self.net = net
        self.loss = loss
        self.opt = opt
        self.dev = dev
        self.checkpoint_epoch = checkpoint_epoch
        self.logger = logger
    
    def data_prepare(self, d):
        (x, y) = d
        x = x.to(self.dev)
        y = y.to(self.dev)
    
    def iterate_dataset(self, dataloader, compare_func, backward, epoch=None):
        pbar = tqdm(range(len(dataloader.dataset)), leave=True)
        total_loss = 0.0
        total_acc = 0.0
        for d in dataloader:
            (x, y) = self.data_prepare(d)
            y_pred = self.net(x)
            loss = self.loss(y_pred, y)
            acc = compare_func(y_pred, y)

            if backward:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            
            postfix = {'loss': loss.item(), 'acc': acc}
            if epoch is not None:
                postfix["epoch"] = epoch

            pbar.set_postfix(postfix)
            pbar.update(len(x))

            total_loss += loss.item()
            total_acc += acc
        pbar.close()

        return (total_loss, total_acc)
    
    def test(self, dataloader, compare_func):
        '''
        compare_func: require f(y_pred, y) and return accuracy
        '''
        self.net.eval()
        (total_loss, total_acc) = self.iterate_dataset(dataloader, compare_func, backward=False)
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        print("======= Test Result =======")
        print(f'avg_loss = {avg_loss}\navg_acc = {avg_acc}')
        print("===========================")

    def train(self, train_dataloader, eval_dataloader, compare_func, eval_epochs, epochs):
        assert self.logger.get_epoch() <= epochs
        for epoch in range(self.logger.get_epoch() + 1, epochs + 1):
            self.net.train()
            (train_total_loss, train_total_acc) = self.iterate_dataset(train_dataloader, compare_func, backward=True, epoch=epoch)    
            train_avg_loss = train_total_loss / len(train_dataloader)
            train_avg_acc = train_total_acc / len(train_dataloader)
            self.logger.log_train(epoch, train_avg_loss, train_avg_acc)
            self.logger.update_avg_loss(train_avg_loss)
            self.logger.add_epoch()

            if (epoch % eval_epochs == 0):
                self.net.eval()
                (eval_total_loss, eval_total_acc) = self.iterate_dataset(eval_dataloader, compare_func, backward=False)    
                eval_avg_loss = eval_total_loss / len(eval_dataloader)
                eval_avg_acc = eval_total_acc / len(eval_dataloader)
                self.logger.log_eval(epoch, eval_avg_loss, eval_avg_acc)

            if (epoch % self.checkpoint_epoch == 0):
                self.logger.save()
            
class YOLO3_Trainer(Trainer):
    def __init__(self, net, loss, opt, dev, logger, checkpoint_epoch):
        super().__init__(net, loss, opt, dev, logger, checkpoint_epoch)
    
    def data_prepare(self, d):
        x = d[0]
        y = d[1:]
        x = x.to(self.dev)
        for i in range(len(y)):
            y[i] = y[i].to(self.dev)
        return (x, y)