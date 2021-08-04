from tqdm import tqdm

class Trainer():
	def __init__(self, net, dataloader, loss, opt, dev, logger, checkpoint_epoch):
		self.net = net
		self.dataloader = dataloader
		self.loss = loss
		self.opt = opt
		self.dev = dev
		self.checkpoint_epoch = checkpoint_epoch
		self.logger = logger
	
	def load(self, json_file):
		pass
	
	def train(self, epochs):
		self.net.train()
		for epoch in range(self.logger.get_epoch() + 1, epochs + 1):
			pbar = tqdm(range(len(self.dataloader.dataset)), leave=True)
			total_loss = 0.0
			for (x, y) in self.dataloader:
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

			avg_loss = total_loss / len(self.dataloader.dataset)
			self.logger.update_avg_loss(avg_loss)
			self.logger.add_epoch()

			if (epoch % self.checkpoint_epoch == 0):
				self.logger.save()