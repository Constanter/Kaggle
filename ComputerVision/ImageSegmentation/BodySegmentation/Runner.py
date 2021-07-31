from catalyst import dl
import torch

class CustomRunner(dl.Runner):
    def predict_batch(self, batch):
        # model inference step
        return self.model(batch[0].to(self.device))
    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x).squeeze(1)
        binar = torch.sigmoid(logits)
        self.batch = {
            "features": x,
            "logits": logits,
            "targets": y,
            "binar": binar,
        }