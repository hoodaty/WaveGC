import torch
from ..metrics.tracker import MetricTracker
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class optimized for WaveGC architecture.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the WaveGC model, compute loss and metrics.
        """
        # 1. Move everything to GPU/Device
        batch = self.move_batch_to_device(batch)

        # 2. Transform Batch (if any augmentation is defined)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        # 3. Prepare Inputs for WaveGCNet
        # Mapping batch keys (from collate_fn) to model args
        model_inputs = {
            'x': batch['x'],
            'eigvs': batch['eigvs'],
            'U': batch['U'],  # Model expects 'Us', collate provides 'U'
            'eigvs_mask': batch['eigvs_mask']
        }

        # Handle Edge Index / Adjacency conditionally
        if 'edge_index' in batch:
            model_inputs['edge_index'] = batch['edge_index']
        else:
            model_inputs['edge_index'] = None

        # 4. Forward Pass
        outputs = self.model(**model_inputs)

        # Store output in batch for loss/metrics
        batch['logits'] = outputs

        # 5. Compute Loss
        # Handle dimensions: Squeeze if labels are [B, 1] but logits are [B]
        if batch['labels'].dim() > 1 and batch['labels'].shape[1] == 1:
            targets = batch['labels'].squeeze(1)
        else:
            targets = batch['labels']

        # Ensure types match (Float for regression, Long for classif)
        # Check config to detect regression task
        if self.config.model.out_dim > 1 and "classification" not in self.config.trainer:
            targets = targets.float()

        loss = self.criterion(outputs, targets)

        # Store loss for logging
        batch['loss'] = loss

        # 6. Backward Pass (Training only)
        if self.is_train:
            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # 7. Update Metrics
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, loss.item())

        for met in metric_funcs:
            metrics.update(met.name, met(outputs, targets))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Required override to prevent NotImplementedError from BaseTrainer.
        """
        pass
