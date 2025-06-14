
import time

import torch
from torch_geometric import seed_everything
from torch_geometric.utils import negative_sampling

from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from .trainer_utils import get_multicass_lp_edge_label, compute_weights_binary_classification, compute_weights_multiclass_classification
from torch_sparse import SparseTensor
from scipy.optimize import minimize, LinearConstraint


def w_norm(coeffs, grad_general_norm, grad_dir_norm, grad_bidir_norm):
    w = coeffs[0] *grad_general_norm + coeffs[1] * grad_dir_norm + coeffs[2] * grad_bidir_norm
    w_l2_norm = np.linalg.norm(w)
    return (w_l2_norm, (2*w.dot(grad_general_norm), 2*w.dot(grad_dir_norm), 2*w.dot(grad_bidir_norm)))


class Trainer():

    def __init__(self,
                 training_objective,
                 dataset,
                 model,
                 logger,
                 device,
                 num_epochs=1000,
                 lr=0.01,
                 log_every_n_steps=10,
                 print_every_n_steps=100,
                 random_seed=42):
        self.dataset = dataset
        self.model = model
        self.logger = logger
        self.seed = random_seed
        self.num_epochs = num_epochs
        self.log_every_n_steps = log_every_n_steps
        self.print_every_n_steps = print_every_n_steps
        self.tro = training_objective
        self.lr = lr

        print(f"Current model: {model.model_name}")
        print(f"Current training objective: {training_objective}")
        # print(f"Current learning rate: {model.optimizer.lr}")
        print(f"Current random seed: {random_seed}")

        seed_everything(random_seed)

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = "cpu"
        self.device = torch.device(device)
        print(f"current device: {device}")

    def train_step(
            self, model, optimizer, criterion, 
            x, edge_index, edge_label, edge_label_index,
            norm = None
        ):
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_index, edge_label_index, sigmoid=False)
        loss = criterion(out, edge_label) if norm is None else norm*criterion(out.type(torch.float64), edge_label)
        loss.backward()
        optimizer.step()
        return loss
    
    def scalarization_loss(self, criterion_gen, criterion_dir, criterion_bidir,
                            norm_gen, norm_dir, norm_bidir, 
                            out, edge_label, 
                           out_dir, edge_label_dir, 
                           out_bidir, edge_label_bidir):
        loss = criterion_gen(out, edge_label.view(-1))
        loss_dir = criterion_dir(out_dir, edge_label_dir.view(-1))
        loss_bidir = criterion_bidir(out_bidir, edge_label_bidir.view(-1))

        tot_loss = norm_gen*loss + norm_dir*loss_dir + norm_bidir*loss_bidir

        return tot_loss, loss, loss_dir, loss_bidir

    def scalarization_train_step(self,
            model, optimizer, criterion_gen, criterion_dir, criterion_bidir,
            norm_gen, norm_dir, norm_bidir,
            x, edge_index, edge_label, edge_label_index,
            edge_label_dir, edge_label_index_dir,
            edge_label_bidir, edge_label_index_bidir
        ):
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if model.model_name in ['gae', 'gravitygae']:
            z = model.encode(x, edge_index)
            out = model.decode(z, edge_label_index, sigmoid=False).view(-1)
            out_dir = model.decode(z, edge_label_index_dir, sigmoid=False).view(-1)
            out_bidir = model.decode(z, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['digae']:
            s, t = model.encode(x, x, edge_index)
            out = model.decode(s, t, edge_label_index, sigmoid=False).view(-1)
            out_dir = model.decode(s, t, edge_label_index_dir, sigmoid=False).view(-1)
            out_bidir = model.decode(s, t, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['mplp']:
            adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
            z = model.encode(x, adj_t)
            out = model.decode(z, adj_t, edge_label_index, sigmoid=False).view(-1)
            out_dir = model.decode(z, adj_t, edge_label_index_dir, sigmoid=False).view(-1)
            out_bidir = model.decode(z, adj_t, edge_label_index_bidir, sigmoid=False).view(-1)

        loss, _, _, _  = self.scalarization_loss(criterion_gen, criterion_dir, criterion_bidir,
                                       norm_gen, norm_dir, norm_bidir, 
                                       out, edge_label,
                                       out_dir, edge_label_dir, 
                                       out_bidir, edge_label_bidir)
        loss.backward()
        optimizer.step()
        return loss
    
    def multiobjective_train_step(self, model, optimizer, 
            criterion_gen, criterion_dir, criterion_bidir,
            x, edge_index, edge_label, edge_label_index,
            edge_label_dir, edge_label_index_dir,
            edge_label_bidir, edge_label_index_bidir):
        
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if model.model_name in ['gae', 'gravitygae']:
            z = model.encode(x, edge_index)
            out = model.decode(z, edge_label_index, sigmoid=False).view(-1)
            # out_dir = model.decode(z, edge_label_index_dir, sigmoid=False).view(-1)
            # out_bidir = model.decode(z, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['digae']:
            s, t = model.encode(x, x, edge_index)
            out = model.decode(s, t, edge_label_index, sigmoid=False).view(-1)
            # out_dir = model.decode(s, t, edge_label_index_dir, sigmoid=False).view(-1)
            # out_bidir = model.decode(s, t, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['mplp']:
            adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
            z = model.encode(x, adj_t)
            out = model.decode(z, adj_t, edge_label_index, sigmoid=False).view(-1)
            # out_dir = model.decode(z, adj_t, edge_label_index_dir, sigmoid=False).view(-1)
            # out_bidir = model.decode(z, adj_t, edge_label_index_bidir, sigmoid=False).view(-1)

        loss = criterion_gen(out, edge_label.view(-1))
        loss.backward()
        grad_general = []
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                grad_general += parameter.grad.reshape(-1).tolist()

        grad_general = np.array(grad_general)
        loss_item = loss.item()
        model.zero_grad()

        if model.model_name in ['gae', 'gravitygae']:
            z = model.encode(x, edge_index)
            # out = model.decode(z, edge_label_index, sigmoid=False).view(-1)
            out_dir = model.decode(z, edge_label_index_dir, sigmoid=False).view(-1)
            # out_bidir = model.decode(z, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['digae']:
            s, t = model.encode(x, x, edge_index)
            # out = model.decode(s, t, edge_label_index, sigmoid=False).view(-1)
            out_dir = model.decode(s, t, edge_label_index_dir, sigmoid=False).view(-1)
            # out_bidir = model.decode(s, t, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['mplp']:
            adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
            z = model.encode(x, adj_t)
            # out = model.decode(z, adj_t, edge_label_index, sigmoid=False).view(-1)
            out_dir = model.decode(z, adj_t, edge_label_index_dir, sigmoid=False).view(-1)
            # out_bidir = model.decode(z, adj_t, edge_label_index_bidir, sigmoid=False).view(-1)
        loss_dir = criterion_dir(out_dir, edge_label_dir.view(-1))
        loss_dir.backward()
        grad_dir = []
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                grad_dir += parameter.grad.reshape(-1).tolist()

        grad_dir = np.array(grad_dir)
        loss_dir_item = loss_dir.item()
        model.zero_grad()

        if model.model_name in ['gae', 'gravitygae']:
            z = model.encode(x, edge_index)
            # out = model.decode(z, edge_label_index, sigmoid=False).view(-1)
            # out_dir = model.decode(z, edge_label_index_dir, sigmoid=False).view(-1)
            out_bidir = model.decode(z, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['digae']:
            s, t = model.encode(x, x, edge_index)
            # out = model.decode(s, t, edge_label_index, sigmoid=False).view(-1)
            # out_dir = model.decode(s, t, edge_label_index_dir, sigmoid=False).view(-1)
            out_bidir = model.decode(s, t, edge_label_index_bidir, sigmoid=False).view(-1)
        elif model.model_name in ['mplp']:
            adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
            z = model.encode(x, adj_t)
            # out = model.decode(z, adj_t, edge_label_index, sigmoid=False).view(-1)
            # out_dir = model.decode(z, adj_t, edge_label_index_dir, sigmoid=False).view(-1)
            out_bidir = model.decode(z, adj_t, edge_label_index_bidir, sigmoid=False).view(-1)
        loss_bidir = criterion_bidir(out_bidir, edge_label_bidir.view(-1))
        loss_bidir.backward()
        grad_bidir = []
        for _, parameter in model.named_parameters():
            if parameter.requires_grad:
                grad_bidir += parameter.grad.reshape(-1).tolist()

        grad_bidir = np.array(grad_bidir)
        loss_bidir_item = loss_bidir.item()
        
        general_length  = np.linalg.norm(grad_general)
        directional_length = np.linalg.norm(grad_dir)
        bidirectional_length = np.linalg.norm(grad_bidir)
        # print("grad gen", grad_general)
        # print("grad dir", grad_dir)
        # print("grad bidir", grad_bidir)
        if general_length != 0:
            grad_general =  grad_general / general_length
        if directional_length != 0:
            grad_dir =  grad_dir / directional_length
        if bidirectional_length != 0:
            grad_bidir =  grad_bidir / bidirectional_length

        res = minimize(w_norm, [0.33,0.34, 0.33], args = (grad_general, grad_dir, grad_bidir), jac = True, bounds = [(0.,1.), (0.,1.), (0.,1.)], constraints = [LinearConstraint(A = [[1,1,1]], lb = 1., ub = 1.)],)
        grad_mo = torch.tensor(res.x[0]*grad_general+res.x[1]*grad_dir+res.x[2]*grad_bidir).to(self.device)
        idx = 0
        for _, par in model.named_parameters():
            if par.requires_grad:
                shape = tuple(par.grad.shape)
                tot_len = np.prod(shape).astype(int) # shape[0]*shape[1]
                par.grad = grad_mo[idx:(idx + tot_len)].reshape(shape).to(torch.float)
                idx += tot_len

        optimizer.step()
        tot_loss = loss+loss_dir+loss_bidir   
        return tot_loss, loss, loss_dir, loss_bidir
        
    def val_step(
            self, model, criterion, 
            x, edge_index, edge_label, edge_label_index,
            norm=None
        ):
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_label_index, sigmoid=False)
            loss = criterion(out, edge_label.view(-1)) if norm is None else norm*criterion(out.type(torch.float64), edge_label.view(-1))
        return loss
    
    def scalarization_val_step(self,
            model, criterion_gen, criterion_dir, criterion_bidir,
            norm_gen, norm_dir, norm_bidir,
            x, edge_index, edge_label, edge_label_index,
            edge_label_dir, edge_label_index_dir,
            edge_label_bidir, edge_label_index_bidir
        ):
        
        model.eval()
        with torch.no_grad():
            if model.model_name in ['gae', 'gravitygae']:
                z = model.encode(x, edge_index)
                out = model.decode(z, edge_label_index, sigmoid=False).view(-1)
                out_dir = model.decode(z, edge_label_index_dir, sigmoid=False).view(-1)
                out_bidir = model.decode(z, edge_label_index_bidir, sigmoid=False).view(-1)
            elif model.model_name in ['digae']:
                s, t = model.encode(x, x, edge_index)
                out = model.decode(s, t, edge_label_index, sigmoid=False).view(-1)
                out_dir = model.decode(s, t, edge_label_index_dir, sigmoid=False).view(-1)
                out_bidir = model.decode(s, t, edge_label_index_bidir, sigmoid=False).view(-1)
            elif model.model_name in ['mplp']:
                adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
                z = model.encode(x, adj_t)
                out = model.decode(z, adj_t, edge_label_index, sigmoid=False).view(-1)
                out_dir = model.decode(z, adj_t, edge_label_index_dir, sigmoid=False).view(-1)
                out_bidir = model.decode(z, adj_t, edge_label_index_bidir, sigmoid=False).view(-1)

        loss, gen_loss, dir_loss, bidir_loss = self.scalarization_loss(criterion_gen, criterion_dir, criterion_bidir,
                                       norm_gen, norm_dir, norm_bidir, 
                                       out, edge_label,
                                       out_dir, edge_label_dir, 
                                       out_bidir, edge_label_bidir)
        return loss, gen_loss, dir_loss, bidir_loss

    def test_step(self, model, x, edge_index, edge_label, edge_label_index):
        model.eval()
        with torch.no_grad():
            if self.tro == 'multiclass':
                y_pred = model(x, edge_index, edge_label_index, sigmoid=True, binary=True)
            else:
                y_pred = model(x, edge_index, edge_label_index, sigmoid=True)
        return y_pred.cpu().numpy(), edge_label.cpu().numpy()
    
    def test(self, x, edge_index, edge_label, edge_label_index):

        pred, y = self.test_step(self.model, x, edge_index, edge_label, edge_label_index)
        return np.array([roc_auc_score(y, pred), average_precision_score(y, pred)])
        
    def train(self):
        print('Training...')
        train_start = time.time()

        self.model.to(self.device)
        self.dataset.to(self.device)

        num_nodes = self.dataset.num_nodes

        # gen_loss, dir_loss, bidir_loss = 1., 1., 1.
        # val_loss = gen_loss+dir_loss+bidir_loss
        # norm_general = gen_loss/val_loss
        # norm_dir = dir_loss/val_loss
        # norm_bidir = bidir_loss/val_loss

        optimizer = self.model.optimizer
        if self.tro == 'baseline':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif self.tro == 'scalarization' or self.tro == 'multiobjective':
            norm_general, weights_general = compute_weights_binary_classification(self.dataset.train_general_pos.edge_label)
            norm_dir, weights_dir = compute_weights_binary_classification(self.dataset.train_directional.edge_label)
            norm_bidir, weights_bidir = compute_weights_binary_classification(self.dataset.train_bidirectional.edge_label)
            # print(f"Loss normalizations: {norm_general}, {norm_dir}, {norm_bidir}")
            criterion_general = torch.nn.BCEWithLogitsLoss(pos_weight=weights_general)
            criterion_dir = torch.nn.BCEWithLogitsLoss(pos_weight=weights_dir)
            criterion_bidir = torch.nn.BCEWithLogitsLoss(pos_weight=weights_bidir)
        elif self.tro == 'multiclass':
            # Do a run of negative sampling to compute weights
            pos_edge_index = self.dataset.train_general_pos.edge_label_index 
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes, pos_edge_index.shape[1])

            edge_label_index = torch.cat([
                pos_edge_index,
                neg_edge_index
            ], dim=1).to(self.device)
            edge_label = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ], dim=0).to(self.device)

            edge_label_class, _ = get_multicass_lp_edge_label(
                        edge_label_index,
                        edge_label.view(-1), 
                        self.device)
            norm, weights = compute_weights_multiclass_classification(edge_label_class) 
            norm = norm.type(torch.float64)
            weights = weights.type(torch.float64)
            print(f"Multiclass norm and weights: {norm} - {weights}")
            criterion = torch.nn.NLLLoss(weight=weights)
        else:
            raise NotImplementedError()
        
        self.model.train()
        for i in range(self.num_epochs):

            epoch_start = time.time()
            x = self.dataset.train_general_pos.x
            edge_index = self.dataset.train_general_pos.edge_index
            edge_label = self.dataset.train_general_pos.edge_label
            pos_edge_index = self.dataset.train_general_pos.edge_label_index 

            neg_edge_index = negative_sampling(pos_edge_index, num_nodes, pos_edge_index.shape[1])
            # print("negative sampling:", neg_edge_index.size())

            edge_label_index = torch.cat([
                pos_edge_index,
                neg_edge_index
            ], dim=1).to(self.device)

            edge_label = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ], dim=0).to(self.device)
            
            if self.tro == 'baseline':
                loss = self.train_step(self.model, optimizer, criterion,
                                    x, edge_index, edge_label, edge_label_index)

            elif self.tro == 'scalarization':
                edge_label_dir = self.dataset.train_directional.edge_label
                edge_label_index_dir = self.dataset.train_directional.edge_label_index 
                
                edge_label_bidir = self.dataset.train_bidirectional.edge_label
                edge_label_index_bidir = self.dataset.train_bidirectional.edge_label_index
                
                loss = self.scalarization_train_step(self.model, optimizer, criterion_general, criterion_dir, 
                                                     criterion_bidir,
                                    norm_general, norm_dir, norm_bidir,
                                    x, edge_index, edge_label, edge_label_index,
                                    edge_label_dir, edge_label_index_dir,
                                    edge_label_bidir, edge_label_index_bidir)
            elif self.tro == 'multiobjective':
                edge_label_dir = self.dataset.train_directional.edge_label
                edge_label_index_dir = self.dataset.train_directional.edge_label_index 
                
                edge_label_bidir = self.dataset.train_bidirectional.edge_label
                edge_label_index_bidir = self.dataset.train_bidirectional.edge_label_index
                
                loss, _, _, _ = self.multiobjective_train_step(self.model, 
                                    optimizer, criterion_general, criterion_dir, criterion_bidir,
                                    x, edge_index, edge_label, edge_label_index,
                                    edge_label_dir, edge_label_index_dir,
                                    edge_label_bidir, edge_label_index_bidir)
                
            elif self.tro == 'multiclass':
                edge_label_class, edge_label_index_class = get_multicass_lp_edge_label(
                    edge_label_index,
                    edge_label.view(-1), 
                    self.device)
                loss = self.train_step(self.model, optimizer, criterion,
                                    x, edge_index, edge_label_class, edge_label_index_class, norm=norm)
            else:
                raise NotImplementedError()
                
            epoch_time = time.time() - epoch_start
            
            # Eval
            if i % self.log_every_n_steps == 0:
                train_score = self.test(self.dataset.train_general_pos.x, self.dataset.train_general_pos.edge_index, edge_label, edge_label_index)
                
                x = self.dataset.val_general.x
                edge_index = self.dataset.val_general.edge_index
                edge_label = self.dataset.val_general.edge_label
                edge_label_index = self.dataset.val_general.edge_label_index
                if self.tro == 'baseline':
                    val_loss = self.val_step(self.model, criterion,
                                        x, edge_index, edge_label, edge_label_index)
                elif self.tro == 'scalarization' or self.tro == 'multiobjective':
                    edge_label_dir = self.dataset.val_directional.edge_label
                    edge_label_index_dir = self.dataset.val_directional.edge_label_index 
                    
                    edge_label_bidir = self.dataset.val_bidirectional.edge_label
                    edge_label_index_bidir = self.dataset.val_bidirectional.edge_label_index
                    val_loss, gen_loss, dir_loss, bidir_loss = self.scalarization_val_step(self.model, criterion_general, criterion_dir, 
                                                     criterion_bidir,
                                        norm_general, norm_dir, norm_bidir,
                                        x, edge_index, edge_label, edge_label_index,
                                        edge_label_dir, edge_label_index_dir,
                                        edge_label_bidir, edge_label_index_bidir)
                if self.tro == 'multiclass':
                    edge_label_class, edge_label_index_class = get_multicass_lp_edge_label(
                        edge_label_index,
                        edge_label.view(-1), 
                        self.device)
                    val_loss = self.val_step(self.model, criterion,
                                        x, edge_index, edge_label_class, edge_label_index_class, norm=norm)
                
                val_score = self.test(self.dataset.val_general.x, self.dataset.val_general.edge_index, self.dataset.val_general.edge_label[:, 0], self.dataset.val_general.edge_label_index)
                test_score = self.test(self.dataset.test_general.x, self.dataset.test_general.edge_index, self.dataset.test_general.edge_label[:, 0], self.dataset.test_general.edge_label_index)
                self.logger.log_metrics(
                    {
                        "train_loss": loss.detach().cpu().item(),
                        "val_loss": val_loss.detach().cpu().item(),
                        "train_roc_auc": train_score[0],
                        "train_avg_prec": train_score[1],
                        "val_general_roc_auc": val_score[0],
                        "val_general_avg_prec": val_score[1],
                        "test_general_roc_auc": test_score[0],
                        "test_general_avg_prec": test_score[1]
                    },
                    step=i,
                    run=self.seed
                )
                
                val_dir_score = self.test(self.dataset.val_directional.x, self.dataset.val_directional.edge_index, self.dataset.val_directional.edge_label[:, 0], self.dataset.val_directional.edge_label_index)
                test_dir_score = self.test(self.dataset.test_directional.x, self.dataset.test_directional.edge_index, self.dataset.test_directional.edge_label[:, 0], self.dataset.test_directional.edge_label_index)
                self.logger.log_metrics(
                    {
                        "val_directional_roc_auc": val_dir_score[0],
                        "val_directional_avg_prec": val_dir_score[1],
                        "test_directional_roc_auc": test_dir_score[0],
                        "test_directional_avg_prec": test_dir_score[1]
                    },
                    step=i,
                    run=self.seed
                )
                val_bidir_score = self.test(self.dataset.val_bidirectional.x, self.dataset.val_bidirectional.edge_index, self.dataset.val_bidirectional.edge_label[:, 0], self.dataset.val_bidirectional.edge_label_index)
                test_bidir_score = self.test(self.dataset.test_bidirectional.x, self.dataset.test_bidirectional.edge_index, self.dataset.test_bidirectional.edge_label[:, 0], self.dataset.test_bidirectional.edge_label_index)
                self.logger.log_metrics(
                    {
                        "val_bidirectional_roc_auc": val_bidir_score[0],
                        "val_bidirectional_avg_prec": val_bidir_score[1],
                        "test_bidirectional_roc_auc": test_bidir_score[0],
                        "test_bidirectional_avg_prec": test_bidir_score[1]
                    },
                    step=i,
                    run=self.seed
                )
                
                cum_time = time.time() - train_start
                self.logger.log_metrics(
                    {
                        "cumulative_time": cum_time,
                        "epoch_time": epoch_time
                    },
                    step=i,
                    run=self.seed
                )

            # Print on screen
            if i % self.print_every_n_steps == 0:
                print(f"loss at epoch {i+1}/{self.num_epochs}: {loss:.3f} - validation loss: {val_loss:.3f}")
                print(f"[GENERAL] - Train score: {train_score} - Val score: {val_score} - Test score: {test_score}")
                print(f"[DIRECTIONAL] - Val score {val_dir_score} - Test score: {test_dir_score}")
                print(f"[BIDIRECTIONAL] - Val score {val_bidir_score} - Test score: {test_bidir_score}")
