import os
import torch 
import horovod.torch as hvd
from utils import remove_partial_mean_with_mask, assert_partial_mean_zero_with_mask
from visualizer import save_xyz_file
from dataset import create_templates_for_linker_generation


class Trainer:
    def __init__(
        self,
        model,
        device,
        epochs,
        analyze_epochs,
        optimizer,
        run,
        loss_type,
        save_path,
        save_prefix,
        n_stability_samples=10, 
    ) -> None:
        self.device = device
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.run = run
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.loss_type = loss_type
        self.analyze_epochs = analyze_epochs
        self.n_stability_samples = n_stability_samples 
    
    def pred(self, dataloader, output_dir, sample_fn=None):

        for data in dataloader:
            uuids = []
            true_names = []
            frag_names = []
            for uuid in data['uuid']:
                uuid = str(uuid)
                uuids.append(uuid)
                true_names.append(f'{uuid}/true')
                frag_names.append(f'{uuid}/frag')
                os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)


            # Removing COM of fragment from the atom coordinates
            h, x, node_mask, frag_mask = data['one_hot'], data['positions'], data['atom_mask'], data['fragment_mask']

            center_of_mass_mask = data['fragment_mask']

            x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
            assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

            # Saving ground-truth molecules
            save_xyz_file(output_dir, h, x, node_mask, true_names)

            # Saving fragments
            save_xyz_file(output_dir, h, x, frag_mask, frag_names)

            # Sampling and saving generated molecules
            for i in range(self.n_stability_samples):
                chain, node_mask = self.sample_chain(data, sample_fn, keep_frames=1)
                x = chain[0][:, :, :3]
                h = chain[0][:, :, 3:]

                pred_names = [f'{uuid}/{i}' for uuid in uuids]
                save_xyz_file(output_dir, h, x, node_mask, pred_names)

    def train(self, train_loader, val_loader, test_loader):
        # self.test_epoch(test_loader)
        for epoch in range(self.epochs):
            print(epoch)
            self.train_epoch(train_loader)
            self.val_epoch(val_loader, epoch)
        # if self.device == 0:
        #     self.test_epoch(test_loader)
    
    def train_epoch(self, loader):
        self.model.train()
        step_outputs = []
        for data in loader:
            self.optimizer.zero_grad()
            output = self._step(data, training=True)       
            output['loss'].backward()
            self.optimizer.step()
            for metric in output.keys():
                self.run.log({f'{metric}/train_step': output[metric]})
            step_outputs.append(output)
        with torch.no_grad():
            for metric in step_outputs[0].keys():
                avg_metric = Trainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/train_epoch': avg_metric})
    
    def val_epoch(self, loader, epoch):
        best_loss = 100
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = Trainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/val': avg_metric})
                if metric == 'loss' and avg_metric < best_loss:
                    best_loss = avg_metric
                    torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_best.ckpt')
                print(f'{metric}/val: {avg_metric}')
            
            if (epoch + 1) % self.analyze_epochs == 0:
                torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_{epoch}.ckpt')

    def test_epoch(self, loader):
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = Trainer.aggregate_metric(step_outputs, metric)
                print(f'{metric}/test: {avg_metric}')
            self.pred(loader, f'{self.save_prefix}_test')
    
    def _step(self, data, training=False):
        l2_loss = self.model(data, training)
        if self.loss_type == 'l2':
            loss, loss_x, loss_h = l2_loss
        else:
            raise NotImplementedError(self.loss_type)
        
        metrics = {
            'loss': loss,
            'loss_x': loss_x,
            'loss_h': loss_h,
        }

        return metrics

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            linker_sizes = data['linker_mask'].sum(1).view(-1).int()
        else:
            linker_sizes = sample_fn(data)

        template_data = create_templates_for_linker_generation(data, linker_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']
        context = fragment_mask
        center_of_mass_mask = fragment_mask

        x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        chain = self.model.sample_chain(
            x=x.to(self.device),
            h=h.to(self.device),
            node_mask=node_mask.to(self.device),
            edge_mask=edge_mask.to(self.device),
            fragment_mask=fragment_mask.to(self.device),
            linker_mask=linker_mask.to(self.device),
            context=context.to(self.device),
            keep_frames=keep_frames,
        )
        return chain, node_mask

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()

