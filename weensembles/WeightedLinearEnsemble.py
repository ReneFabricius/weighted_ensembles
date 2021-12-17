import numpy as np
import torch
import pickle
import pandas as pd
from scipy.stats import normaltest
from timeit import default_timer as timer

from weensembles.CouplingMethods import coup_picker
from weensembles.CombiningMethods import comb_picker
from weensembles.utils import logit, pairwise_accuracies, pairwise_accuracies_penultimate


class WeightedLinearEnsemble:
    def __init__(self, c=0, k=0, device=torch.device("cpu"), dtp=torch.float32):
        """
        Trainable ensembling of classification posteriors.
        :param c: Number of classifiers
        :param k: Number of classes
        :param device: Torch device to use for computations
        :param dtp: Torch datatype to use for computations
        """
        self.dev_ = device
        self.dtp_ = dtp
        self.logit_eps_ = 1e-5
        self.c_ = c
        self.k_ = k
        self.coefs_ = torch.zeros(k, k, c + 1, device=self.dev_, dtype=self.dtp_)
        self.cls_models_ = [[None for _ in range(k)] for _ in range(k)]
        self.pvals_ = None
        self.trained_on_penultimate_ = None
        self.combine_probs_ = False

    @torch.no_grad()
    def fit(self, MP, tar, combining_method,
            verbose=0, test_normality=False, penultimate=True, MP_val=None, tar_val=None):
        """
        Trains linear classifier on logits of pairwise probabilities for every pair of classes
        :param combining_method: linear classifier to use.
        :param MP: c x n x k tensor of constituent classifiers outputs.
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param tar: n tensor of sample labels
        :param verbose: print more detailed output
        :param test_normality: test normality of lda predictors for each class in each class pair
        :param penultimate: Whether outputs of classifiers in parameter MP are from penultimate layer(logits) or from softmax.
        :param MP_val: Validation set used for hyperparameter sweep. Required if combining_method.req_val is True.
        :param tar_val: Validation set targets. Required if combining_method.req_val is True. 
        :return:
        """

        print("Starting fit")
        start = timer()
        comb_m = comb_picker(combining_method)
        if comb_m is None:
            print("Unknown combining method {} selected".format(combining_method))
            return 1
        
        inc_val = comb_m.req_val
        if inc_val and (MP_val is None or tar_val is None):
            print("MP_val and tar_val are required for combining method {}".format(comb_m.__name__))
            return 1
        
        self.combine_probs_ = comb_m.combine_probs
        
        num = self.k_ * (self.k_ - 1) // 2      # Number of pairs of classes
        if test_normality:
            self.pvals_ = torch.zeros(num, 2, self.c_, device=torch.device("cpu"), dtype=self.dtp_)
        print_step = num // 100

        if not penultimate:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: {} samples with non unit sum of supports found, \
                      performing softmax".format(num_non_one))
                MP = self.softmax_supports(MP)
            if inc_val:
                num_non_one_val = torch.sum(torch.abs(torch.sum(MP_val, dim=2) - 1.0) > self.logit_eps_).item()
                if num_non_one_val > 0:
                    print("Warning: {} samples with non unit sum of supports found in validation set, \
                          performing softmax".format(num_non_one))
                    MP_val = self.softmax_supports(MP_val)
        
        if comb_m.fit_pairwise:
            pi = 0
            for fc in range(self.k_):
                for sc in range(fc + 1, self.k_):
                    if print_step > 0 and pi % print_step == 0:
                        print("Fit progress {}%".format(pi // print_step), end="\r")

                    # c x n tensor containing True for samples belonging to classes fc, sc
                    SamM = (tar == fc) + (tar == sc)
                    if inc_val:
                        SamM_val = (tar_val == fc) + (tar_val == sc)
                    if penultimate:
                        # c x s x 1 tensor, where s is number of samples in classes fc and sc.
                        # Tensor contains support of networks for class fc minus support for class sc
                        SS = MP[:, SamM][:, :, fc] - MP[:, SamM][:, :, sc]

                        # s x c tensor of logit supports of k networks for class fc against class sc for s samples
                        X = SS.squeeze().transpose(0, 1)

                        if inc_val:
                            SS_val = MP_val[:, SamM_val][:, :, fc] - MP_val[:, SamM_val][:, :, sc]
                            X_val = SS_val.squeeze().transpose(0, 1)
                    else:
                        # c x s x 2 tensor, where s is number of samples in classes fc and sc.
                        # Tensor contains supports of networks for classes fc, sc for samples belonging to fc, sc.
                        SS = MP[:, SamM][:, :, [fc, sc]]
                        # c x s tensor containing p_fc,sc pairwise probabilities for above mentioned samples
                        PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                        LI = logit(PWP, self.logit_eps_)
                        # s x c tensor of logit supports of k networks for class fc against class sc for s samples
                        X = LI.transpose(0, 1)
                        
                        if inc_val:
                            SS_val = MP_val[:, SamM_val][:, :, [fc, sc]]
                            PWP_val = torch.true_divide(SS_val[:, :, 0], torch.sum(SS_val, 2) + (SS_val[:, :, 0] == 0))
                            LI_val = logit(PWP_val, self.logit_eps_)
                            X_val = LI_val.transpose(0, 1)
                            
                    # Prepare targets
                    y = tar[SamM]
                    mask_fc = (y == fc)
                    mask_sc = (y == sc)
                    y[mask_fc] = 1
                    y[mask_sc] = 0
                    
                    if inc_val:
                        y_val = tar_val[SamM_val]
                        mask_fc_val = (y_val == fc)
                        mask_sc_val = (y_val == sc)
                        y_val[mask_fc_val] = 1
                        y_val[mask_sc_val] = 0

                    if test_normality:
                        # Test normality of predictors
                        #fc_pval = torch.tensor([normal_ad(X[mask_fc][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                        #sc_pval = torch.tensor([normal_ad(X[mask_sc][:, ci].numpy(), 0)[1] for ci in range(self.c_)])
                        fc_pval = torch.tensor(normaltest(X[mask_fc], 0)[1])
                        sc_pval = torch.tensor(normaltest(X[mask_sc], 0)[1])
                        self.pvals_[pi, 0, :] = fc_pval
                        self.pvals_[pi, 1, :] = sc_pval
                        if verbose > 0:
                            print("P-values of normality test for class " + str(fc))
                            print(str(fc_pval))
                            print("P-values of normality test for class " + str(sc))
                            print(str(sc_pval))

                    if inc_val:
                        clf = comb_m(X=X, y=y, val_X=X_val, val_y=y_val, verbose=verbose)
                    else:
                        clf = comb_m(X=X, y=y, verbose=verbose)
                        
                    self.cls_models_[fc][sc] = clf
                    self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                                                        torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

                    if verbose > 1:
                        pwacc = pairwise_accuracies(SS, y)
                        print("Training pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                                "\n\tpairwise accuracies: " + str(pwacc) +
                                "\n\tchosen coefficients: " + str(clf.coef_) +
                                "\n\tintercept: " + str(clf.intercept_))

                        print("\tcombined accuracy: " + str(clf.score(X.cpu(), y.cpu())))

                    pi += 1

        else:
            if inc_val:
                clf = comb_m(X=MP, y=tar, val_X=MP_val, val_y=tar_val, verbose=verbose)
            else:
                clf = comb_m(X=MP, y=tar, verbose=verbose)
            
            for fc in range(self.k_):
                for sc in range(fc + 1, self.k_):
                    self.cls_models_[fc][sc] = clf
                    self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                                                        torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

        if test_normality:
            blw_5 = torch.sum(self.pvals_ < 0.05)
            blw_1 = torch.sum(self.pvals_ < 0.01)
            print("Number of classes with normality pval below 5% " + str(blw_5.item()))
            print("Number of classes with normality pval below 1% " + str(blw_1.item()))

        end = timer()
        self.trained_on_penultimate_ = penultimate
        print("Fit finished in " + str(end - start) + " s")
        return 0
    
    @torch.no_grad()
    def predict_proba(self, MP, coupling_method, verbose=0, output_R=False, batch_size=None):
        """
        Combines outputs of constituent classifiers using all classes.
        :param batch_size: batch size for coupling method, default None - single batch
        :param output_R: whether to output as a second return the R matrices which enter into coupling method
        :param verbosity: Level of detailed output.
        :param MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param coupling_method: coupling method to use
        :return: n x k tensor of combined posteriors
        """
        coup_m = coup_picker(coupling_method)
        if coup_m is None:
            print("Unknown coupling method {} selected".format(coupling_method))
            return 1
        
        if verbose > 0:
            print("Starting predict proba, coupling method {}".format(coup_m.__name__))
        if self.trained_on_penultimate_ is None:
            print("Ensemble not trained")
            return

        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_
        p_probs = torch.zeros(n, k, k, dtype=self.dtp_)

        if not self.trained_on_penultimate_:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                if not self.trained_on_penultimate_:
                    # Obtains fc and sc probabilities for all classes
                    SS = MP[:, :, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
                    # Computes p_ij pairwise probabilities for above mentioned samples
                    PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                    LI = logit(PWP, self.logit_eps_)
                    X = LI.transpose(0, 1).cpu()
                else:
                    SS = MP[:, :, fc] - MP[:, :, sc]
                    if SS.dim() == 3:
                        SS = SS.squeeze()
                    X = SS.transpose(0, 1)

                PP = self.cls_models_[fc][sc].predict_proba(X.detach().cpu())
                p_probs[:, sc, fc] = torch.from_numpy(PP[:, 0])
                p_probs[:, fc, sc] = torch.from_numpy(PP[:, 1])

        end = timer()
        if verbose > 0:
            print("Predict proba finished in " + str(end - start) + " s")

        R_dev_dtp = p_probs.to(device=self.dev_, dtype=self.dtp_)

        b_size = batch_size if batch_size is not None else n
        prob_batches = []
        for start_ind in range(0, n, b_size):
            batch_probs = coup_m(R_dev_dtp[start_ind:(start_ind + b_size), :, :], verbose=verbose)
            prob_batches.append(batch_probs)

        probs = torch.cat(prob_batches, dim=0)

        if output_R:
            return probs, R_dev_dtp

        return probs

    @torch.no_grad()
    def test_pairwise(self, MP, tar):
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                # Obtains fc and sc probabilities for samples belonging to those classes
                SS = MP[:, (tar == fc) + (tar == sc)][:, :, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
                # Computes p_ij pairwise probabilities for above mentioned samples
                PWP = torch.true_divide(SS[:, :, 0], torch.sum(SS, 2) + (SS[:, :, 0] == 0))
                LI = logit(PWP, self.logit_eps_)
                X = LI.transpose(0, 1).cpu()
                y = tar[(tar == fc) + (tar == sc)]
                mask_fc = (y == fc)
                mask_sc = (y == sc)
                y[mask_fc] = 1
                y[mask_sc] = 0

                pwacc = pairwise_accuracies(SS, y)
                print(
                    "Testing pairwise accuracies for classes: " + str(fc) + ", " + str(sc) +
                    "\n\tpairwise accuracies: " + str(pwacc) +
                    "\n\tchosen coefficients: " + str(self.cls_models_[fc][sc].coef_) +
                    "\n\tintercept: " + str(self.cls_models_[fc][sc].intercept_))

                print("\tcombined accuracy: " + str(self.cls_models_[fc][sc].score(X, y)))

    @torch.no_grad()
    def predict_proba_topl(self, MP, l, coupling_method, verbose=0):
        """
        Combines outputs of constituent classifiers using only those classes, which are among the top l most probable
        for some constituent classifier.
        :param MP: MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param l: how many most probable classes for each constituent classifier to consider
        :param coupling_method: coupling method to use
        :param verbosity: Level of detail in printed output.
        :return: n x k tensor of combined posteriors
        """
        coup_m = coup_picker(coupling_method)
        if coup_m is None:
            print("Unknown coupling method {} selected".format(coupling_method))
            return 1
        if verbose > 0:
            print("Starting predict proba topl")
        if self.trained_on_penultimate_ is None:
            print("Ensemble not trained")
            return
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        if not self.trained_on_penultimate_:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

        ps = torch.zeros(n, k, device=self.dev_, dtype=self.dtp_)
        # Every sample may have different set of top classes, so we process them one by one
        print_step = n // 100
        for ni in range(n):
            if verbose > 0 and ni % print_step == 0:
                print("Predicting proba topl progress " + str(ni//print_step) + "%", end="\r")

            val, ind = torch.topk(MP[:, ni, :], l, dim=1)
            Ti = sorted(list(set(ind.flatten().tolist())))
            tcc = len(Ti)
            p_probs = torch.zeros(1, tcc, tcc, device=self.dev_, dtype=self.dtp_)
            for fci, fc in enumerate(Ti):
                for sci, sc in enumerate(Ti[fci + 1:]):
                    if not self.trained_on_penultimate_:
                        # Obtains fc and sc probabilities for current sample
                        SS = MP[:, ni, [fc, sc]].to(device=self.dev_, dtype=self.dtp_)
                        # Computes p_ij pairwise probabilities for above mentioned samples
                        PWP = torch.true_divide(SS[:, 0], torch.sum(SS, 1) + (SS[:, 0] == 0))
                        LI = logit(PWP, self.logit_eps_)
                        X = LI.T.unsqueeze(0).cpu()
                    else:
                        SS = MP[:, ni, fc] - MP[:, ni, sc]
                        X = SS.T.unsqueeze(0)
                    PP = self.cls_models_[fc][sc].predict_proba(X)
                    p_probs[:, fci + 1 + sci, fci] = torch.from_numpy(PP[:, 0])
                    p_probs[:, fci, fci + 1 + sci] = torch.from_numpy(PP[:, 1])

            sam_ps = coup_m(p_probs.to(device=self.dev_, dtype=self.dtp_), verbose=verbose)
            ps[ni, Ti] = sam_ps.squeeze()

        end = timer()
        if verbose > 0:
            print("Predict proba topl finished in " + str(end - start) + " s")

        return ps

    def predict_proba_topl_fast(self, MP, l, coupling_method, batch_size=None, verbose=0):
        """
        Better optimized version of predict_proba_topl
        Combines outputs of constituent classifiers using only those classes, which are among the top l most probable
        for some constituent classifier.
        :param MP: MP: c x n x k tensor of constituent classifiers posteriors
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :param l: how many most probable classes for each constituent classifier to consider
        :param coupling_method: coupling method to use
        :param batch_size: if not none, the size of the batches to use
        :return: n x k tensor of combined posteriors
        """
        if verbose > 0:
            print("Starting predict proba topl fast")
        coup_m = coup_picker(coupling_method)
        if coup_m is None:
            print("Unknown coupling method {} selected".format(coupling_method))
            return 1
 
        if self.trained_on_penultimate_ is None:
            print("Ensemble not trained")
            return
        start = timer()
        c, n, k = MP.size()
        assert c == self.c_
        assert k == self.k_

        if not self.trained_on_penultimate_:
            num_non_one = torch.sum(torch.abs(torch.sum(MP, dim=2) - 1.0) > self.logit_eps_).item()
            if num_non_one > 0:
                print("Warning: " + str(num_non_one) +
                      " samples with non unit sum of supports found, performing softmax")
                MP = self.softmax_supports(MP)

        b_size = batch_size if batch_size is not None else n
        ps_list = []

        for start_ind in range(0, n, b_size):
            curMP = MP[:, start_ind:(start_ind + b_size), :].to(device=self.dev_, dtype=self.dtp_)
            curn = curMP.shape[1]

            # ind is c x n x l tensor of top l indices for each sample in each network output
            val, ind = torch.topk(curMP, l, dim=2)
            M = torch.zeros(curMP.shape, dtype=torch.bool, device=self.dev_)
            # place true in positions of top probs
            # c x n x k tensor
            M.scatter_(2, ind, True)
            # combine selections over c inputs
            # n x k tensor containing for each sample a mask of union of top l classes from each constituent classifier
            M = torch.sum(M, dim=0, dtype=torch.bool)
            # zeroe lower values
            # c x n x k tensor
            MPz = curMP * M
            # n x c x k tensor
            MPz.transpose_(0, 1)
            ps = torch.zeros(curn, k, device=self.dev_, dtype=self.dtp_)
            # Selected class counts for every n
            NPC = torch.sum(M, 1).squeeze()
            # goes over possible numbers of classes in union of top l classes from each constituent classifier
            for pc in range(l, l * c + 1):
                # Pick those samples which have pc classes in the union
                # pcn x c x k tensor
                pcMPz = MPz[NPC == pc]
                # Pick pc-class masks
                # pcn x k tensor
                pcM = M[NPC == pc]
                # Number of samples with pc classes in the union
                pcn = pcM.shape[0]
                if pcn == 0:
                    continue

                # One dimensional tensors of row and column indices of elements in upper triangle of pc x pc matrix,
                # ordered from left to right from top to bottom
                pcIMR = torch.tensor([], dtype=torch.long, device=self.dev_)
                pcIMC = torch.tensor([], dtype=torch.long, device=self.dev_)
                for r in range(pc):
                    pcIMR = torch.cat((pcIMR, torch.tensor([r], dtype=torch.long, device=self.dev_).repeat(pc - r - 1)))
                    pcIMC = torch.cat((pcIMC, torch.arange(r + 1, pc, device=self.dev_)))

                # pcn x pc tensor containing for each of the pcn samples indices of classes belonging to the union
                pcMi = torch.nonzero(pcM, as_tuple=False)[:, 1].view(pcM.shape[0], pc)
                # Only the values for classes in the union
                # pcn x c x pc tensor
                pcMPp = pcMPz.gather(2, pcMi.unsqueeze(1).expand(pcn, c, pc))

                # For every pcn and c contains values of top right triangle of matrix,
                # formed from supports expanded over columns, from left to right, from top to bottom
                # pcn x c x pc * (pc - 1) // 2 tensor
                pcMPpR = pcMPp[:, :, pcIMR]
                # For every pcn and c contains values of top right triangle of transposed matrix,
                # formed from supports expanded over columns, from left to right, from top to bottom
                # pcn x c x pc * (pc - 1) // 2 tensor
                pcMPpC = pcMPp[:, :, pcIMC]
                if not self.trained_on_penultimate_:
                    # For every pcn and c contains top right triangle of pairwise probs p_ij
                    # pcn x c x pc * (pc - 1) // 2 tensor
                    pcPWP = pcMPpR / (pcMPpR + pcMPpC)

                    # logit pairwise probs
                    # pcn x c x pc * (pc - 1) // 2 tensor
                    pcLI = logit(pcPWP, self.logit_eps_)
                else:
                    # compute instead as a difference of supports
                    # pcn x c x pc * (pc - 1) // 2 tensor
                    pcLI = pcMPpR - pcMPpC

                # Flattened logits in order of dimensions: pcn; pc x pc top right triangle by rows; c
                pcLIflat = pcLI.transpose(1, 2).flatten()

                # Number of values in top right triangle
                val_ps = pc * (pc - 1) // 2
                # kxk matrix row indexes of values in pcLIflat without considering c sources
                I1woc = pcMi[:, pcIMR]
                # kxk matrix row indexes of values in pcLIflat
                I1 = I1woc.repeat_interleave(c)
                # kxk matrix column indexes of values in pcLIflat without considering c sources
                I2woc = pcMi[:, pcIMC]
                # kxk matrix column indexes of values in pcLIflat
                I2 = I2woc.repeat_interleave(c)
                # source indexes of values in pcLIflat
                I3 = torch.arange(c, device=self.dev_).repeat(pcn * val_ps)

                # Extract linear coefficients
                Ws = self.coefs_[I1, I2, I3]
                Bs = self.coefs_[I1woc.flatten(), I2woc.flatten(), c]

                # Apply linear predict_proba
                pcLC = pcLIflat * Ws
                if self.combine_probs_:
                    pcLC_prob = 1 / (1 + torch.exp(-pcLC))
                    CPWP = torch.mean(pcLC_prob.view(pcn * val_ps, c), dim=1)
                else:
                    pcDEC = torch.sum(pcLC.view(pcn * val_ps, c), 1) + Bs
                    CPWP = 1 / (1 + torch.exp(-pcDEC))

                # Build dense matrices of pairwise probabilities disregarding original positions in all-class setting
                dI0 = torch.arange(0, pcn, device=self.dev_, dtype=self.dtp_).repeat_interleave(val_ps)
                dI1 = pcIMR.repeat(pcn)
                dI2 = pcIMC.repeat(pcn)

                I = torch.cat((dI0.unsqueeze(0), dI1.unsqueeze(0), dI2.unsqueeze(0)), 0)
                DPS = torch.sparse_coo_tensor(I, CPWP, (pcn, pc, pc), device=self.dev_, dtype=self.dtp_).to_dense()
                It = torch.cat((dI0.unsqueeze(0), dI2.unsqueeze(0), dI1.unsqueeze(0)), 0)
                DPSt = torch.sparse_coo_tensor(It, 1 - CPWP, (pcn, pc, pc), device=self.dev_, dtype=self.dtp_).to_dense()
                # DPS now should contain pairwise probabilities
                DPS = DPS + DPSt

                pcPS = coup_m(DPS, verbose=verbose)

                # resulting posteriors for samples with pc picked classes
                ps_cur = torch.zeros(pcn, k, device=self.dev_, dtype=self.dtp_)
                row_mask = (NPC == pc)
                ps_cur[M[row_mask]] = torch.flatten(pcPS)
                # Insert current results into complete tensor of posteriors
                ps[row_mask] = ps_cur

            ps_list.append(ps)

        ps_full = torch.cat(ps_list, dim=0)
        end = timer()
        if verbose > 0:
            print("Predict proba topl fast finished in " + str(end - start) + " s")

        return ps_full

    @torch.no_grad()
    def save(self, file, verbose=0):
        """
        Save trained ensemble into a file.
        :param file: file to save the models to
        :return:
        """
        if verbose > 0:
            print("Saving ensemble into file: " + str(file))
        dump_dict = {"models": self.cls_models_, "on_penult": self.trained_on_penultimate_, "comb_prob": self.combine_probs_}
        with open(file, 'wb') as f:
            pickle.dump(dump_dict, f)

    @torch.no_grad()
    def load(self, file, verbose=0):
        """
        Load trained ensemble from a file.
        :param file: File to load the ensemble from.
        :return:
        """
        if verbose > 0:
            print("Loading models from file: " + str(file))
        with open(file, 'rb') as f:
            dump_dict = pickle.load(f)
            self.cls_models_ = dump_dict["models"]
            self.trained_on_penultimate_ = dump_dict["on_penult"]
            self.combine_probs_ = dump_dict["comb_prob"] if "comb_prob" in dump_dict else False

        self.k_ = len(self.cls_models_)
        if self.k_ > 0:
            self.c_ = len(self.cls_models_[0][1].coef_[0])

        self.coefs_ = torch.zeros(self.k_, self.k_, self.c_ + 1, device=self.dev_, dtype=self.dtp_)

        for fc in range(len(self.cls_models_)):
            for sc in range(fc + 1, len(self.cls_models_[fc])):
                clf = self.cls_models_[fc][sc]
                # Compatibility hack
                if type(clf).__name__ == "Averager":
                    if not hasattr(clf, "combine_probs_"):
                        setattr(clf, "combine_probs_", False)
                        
                self.coefs_[fc, sc, :] = torch.cat((torch.tensor(clf.coef_, device=self.dev_, dtype=self.dtp_).squeeze(),
                                                    torch.tensor(clf.intercept_, device=self.dev_, dtype=self.dtp_)))

    @torch.no_grad()
    def save_coefs_csv(self, file):
        """
        Save linear coefficients into a csv file.
        :param file: file to save the coefficients to
        :return:
        """
        Ls = [None] * ((self.k_ * (self.k_ - 1)) // 2)
        li = 0
        cols = ["i", "j"] + ["coef" + str(k) for k in range(self.c_)] + ["interc"]
        for i in range(self.k_):
            for j in range(i + 1, self.k_):
                cfs = [[i, j] + self.coefs_[i, j].tolist()]
                Ls[li] = pd.DataFrame(cfs, columns=cols)
                li += 1

        df = pd.concat(Ls, ignore_index=True)
        df.to_csv(file, index=False)

    @torch.no_grad()
    def save_pvals(self, file):
        """
        Save normality test p-values into a file, if these were computed during the training.
        :param file: file to save the p-values to
        :return:
        """
        print("Saving pvals into file: " + str(file))
        if self.pvals_ is not None:
            np.save(file, self.pvals_)
        else:
            print("P-values not computed")

    @torch.no_grad()
    def set_averaging_weights(self):
        """
        Set the lda weights equal to one.
        :return:
        """
        for fc in range(self.k_):
            for sc in range(fc + 1, self.k_):
                self.coefs_[fc, sc, :] = torch.tensor([1]*self.c_ + [0])

    @torch.no_grad()
    def softmax_supports(self, MP):
        """
        Performs softmax on posteriors
        :param MP: c x n x k tensor of class supports
        c - number of constituent classifiers, n - number of training samples, k - number of classes
        :return:c x n x k tensor of posteriors
        """
        return torch.nn.Softmax(dim=-1)(MP)
