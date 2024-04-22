import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader

from models.cs236781.train_results import FitResult, BatchResult, EpochResult
from typing import List, NamedTuple

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer,scheduler, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        post_epoch_fn=None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        prev_loss = None
        best_loss = None
        epochs_without_improvement = 0
        starting_epoch=0
        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])
                self.optimizer.load_state_dict(saved_state["optimizer_state_dict"])
                self.scheduler.load_state_dict(saved_state["scheduler_state_dict"])
                starting_epoch = saved_state["epoch"] + 1
        for epoch in range(starting_epoch,num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            
            cur_train_loss, cur_train_acc = self.train_epoch(dl_train, **kw)
            cur_test_loss, cur_test_acc = self.test_epoch(dl_test, **kw)
            def get_mean(li):
                return torch.tensor(li).mean().item()
            cur_train_loss = get_mean(cur_train_loss)
            cur_test_loss = get_mean(cur_test_loss)
            train_loss.append(cur_train_loss)
            test_loss.append(cur_test_loss)
            train_acc.append(cur_train_acc)
            test_acc.append(cur_test_acc)
            
            
            is_best = False
            if best_acc is None or cur_test_acc > best_acc:
                best_acc = cur_test_acc
                
                is_best = True
                save_checkpoint = True and (checkpoints != None)
                
                epochs_without_improvement = 0
                best_loss = cur_test_loss
                
            else:
                epochs_without_improvement += 1  
                
            
            class EpochResult(NamedTuple):
                loss: float
                accuracy: float
            train_result = EpochResult(cur_train_loss, cur_train_acc)
            test_result = EpochResult(cur_test_loss, cur_test_acc)
            self.scheduler.step(- cur_test_loss)
            # ========================
            # Save model checkpoint if requested
            if save_checkpoint and is_best and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    epoch=epoch
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch+1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

            if early_stopping != None and epochs_without_improvement >= early_stopping:
                break
            
        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class PEARTrainer(Trainer):

    def __init__(self, model, loss_fn, optimizer,scheduler, device="cpu", mask_lr=None):
        model = model.to(device)
        super().__init__(model, loss_fn, optimizer,scheduler, device)
        self.mask_lr = mask_lr
        # self.flag = True


    def train_batch(self, batch) -> BatchResult:
        
        x, y = batch
        #orig_x = x
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)
        y = y.unsqueeze(1)
        model_out = self.model(x)
        loss = self.loss_fn(model_out, y)
        self.optimizer.zero_grad()
        loss.backward()
        # if self.flag:
        #     with torch.no_grad():
        #         self.flag = False
        #         import os
        #         orig_x = orig_x.to(self.device)
        #         orig_x = self.model.subsample(orig_x)
        #         args = [("./results/label.pth", y),
        #             ("./results/reconstructed.pth", model_out),
        #             ("./results/input.pth", orig_x),
        #             ("./results/input2.pth", x),
        #             ("./results/mask.pth", self.model.subsample.mask),
        #             ("./results/bin_mask.pth", self.model.subsample.binary_mask),
        #             ("./results/model.pth", self.model.state_dict()),
        #         ]
        #         def saveImgs(path, to_save):
        #             if not os.path.exists(path):
        #                 os.makedirs(os.path.dirname(path), exist_ok=True)
        #             torch.save(to_save, path)
        #         for arg in args:
        #             saveImgs(*arg)
        self.optimizer.step()
        if self.model.learn_mask: 
            self.model.subsample.mask_grad(self.mask_lr)
        return BatchResult(loss.item(), - loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        y = y.to(self.device)
        y = y.unsqueeze(1)
        
        loss = None
        
        with torch.no_grad():
            model_out = self.model(x)
            loss = self.loss_fn(model_out, y)
            

        return BatchResult(loss.item(), - loss.item())


