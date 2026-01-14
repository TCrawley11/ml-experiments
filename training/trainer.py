import torch

class Trainer():
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

    def calc_loss_batch(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            input=logits.flatten(0,1),
            target=target_batch.flatten()
        )
        return loss

    def calc_loss_loader(self, dataloader, num_batches=None):
        total_loss = 0
        if len(dataloader) == 0:
            return float("nan")
        elif num_batches == None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))

        for i, (input_batch, target_batch) in enumerate(dataloader):
            if i < num_batches:
                loss = self.calc_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch
                )
                total_loss += loss.item()
                print(f"total loss: {total_loss} / batch num: {i}")
            else:
                break

            return total_loss / num_batches
        