from tqdm import tqdm

class Trainer():
	def __init__(self, net, dataloader, loss, opt, dev, checkpoint_epoch=None, checkpoint_dir=None):
		self.net = net
		self.dataloader = dataloader
		self.loss = loss
		self.opt = opt
		self.dev = dev
		if checkpoint_epoch is not None:
			assert checkpoint_dir is not None
			self.is_checkpoint = True
		else:
			self.is_checkpoint = False
		self.checkpoint_epoch = checkpoint_epoch
		self.checkpoint_dir = checkpoint_dir
	
	def load(self, json_file):
		pass
	
	def train(self, epochs):
		self.net.train()
		for epoch in range(1, epochs + 1):
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

				total_loss += loss.item()
			pbar.close()
			avg_loss = total_loss / len(self.dataloader.dataset)