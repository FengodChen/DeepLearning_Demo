from tqdm import tqdm

class Trainer():
	def __init__(self, net, loss, opt, dev, logger, checkpoint_epoch):
		self.net = net
		self.loss = loss
		self.opt = opt
		self.dev = dev
		self.checkpoint_epoch = checkpoint_epoch
		self.logger = logger
	
	def train(self, dataloader, epochs):
		self.net.train()
		assert self.logger.get_epoch() <= epochs
		for epoch in range(self.logger.get_epoch() + 1, epochs + 1):
			pbar = tqdm(range(len(dataloader.dataset)), leave=True)
			total_loss = 0.0
			for (x, y) in dataloader:
				x = x.to(self.dev)
				y = y.to(self.dev)
				y_pred = self.net(x)
				loss = self.loss(y_pred, y)
				loss.backward()
				self.opt.step()
				self.opt.zero_grad()

				pbar.set_postfix({'loss': loss.item(), "epoch": epoch})
				pbar.update(len(x))
				self.logger.update_loss(loss.item())

				total_loss += loss.item()
			pbar.close()

			avg_loss = total_loss / len(dataloader)
			self.logger.update_avg_loss(avg_loss)
			self.logger.add_epoch()

			if (epoch % self.checkpoint_epoch == 0):
				self.logger.save()
	
	def eval(self, dataloader, compare_func):
		'''
		compare_func: require f(y_pred, y) and return accuracy
		'''
		self.net.eval()
		pbar = tqdm(range(len(dataloader.dataset)), leave=True)
		total_loss = 0.0
		total_acc = 0.0
		for (x, y) in dataloader:
			x = x.to(self.dev)
			y = y.to(self.dev)
			y_pred = self.net(x)
			loss = self.loss(y_pred, y)
			acc = compare_func(y_pred, y)

			pbar.set_postfix({'loss': loss.item(), 'acc': acc})
			pbar.update(len(x))

			total_loss += loss.item()
			total_acc += acc
		pbar.close()

		avg_loss = total_loss / len(dataloader)
		avg_acc = total_acc / len(dataloader)

		print("======= Eval Result =======")
		print(f'avg_loss = {avg_loss}\navg_acc = {avg_acc}')
		print("===========================")
