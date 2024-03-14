import numpy as np
import math
import gudhi as gd
import torch
import torch.nn.functional as F

"""
Consists of loss functions who able to back-propagation.
The Name of a method starts with 'get_'
"""


def get_loss(outputs, labels, run_cfg, mode: str = 'ce', map=None):
    """ Compute cross entropy loss with/without class_weight """

    if mode.lower() == 'ce':
        if isinstance(outputs, tuple):
            loss = 0
            for i, o in enumerate(outputs):
                loss += run_cfg.feature_weight[i] * F.cross_entropy(o, labels, weight=run_cfg.class_weight)
            loss /= sum(run_cfg.feature_weight)
        else:
            if map is not None:
                loss = torch.mean(F.cross_entropy(outputs, labels, reduction='none') * map)
            else:
                loss = F.cross_entropy(outputs, labels)  # Equivalent to nn.CrossEntropyLoss

    elif mode.lower() == 'topo':
        # start = time.perf_counter()
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = multi_topo_loss(outputs=outputs, labels=labels)
        # print('elapsed:', time.perf_counter() - start)
    else:
        loss = 0
    return loss


def get_amp_map(signal):
    """
    Expected shape of signal is [B, 1, len].
    Return amplitude map with [B, 1, len] shape.
    """
    len_signal = signal.shape[-1]
    pad_tensor = torch.nn.functional.pad(signal, (30, 30), mode='replicate')
    amp_map = F.max_pool1d(pad_tensor, 16) + F.max_pool1d(-pad_tensor, 16)
    amp_map_smooth = F.interpolate(amp_map, len_signal + 60, mode='linear', align_corners=True)[:, :, 30:-30]
    amp_map_norm = amp_map_smooth / (amp_map_smooth.max(-1, keepdims=True)[0] + 1e-8)
    return amp_map_norm


def multi_topo_loss(outputs: torch.tensor, labels: torch.tensor):
    # def topo_ch_loop(likelihood, ground_truth):
    #     loss = 0
    #     for i_ch in range(likelihood.shape[0]):
    #         if i_ch == 0: continue
    #         loss += topo_loss(likelihood[i_ch], ground_truth == i_ch, topo_size=900) / (likelihood.shape[0] - 1)
    #     return loss
    #
    # return np.array(list(map(topo_ch_loop, outputs, labels))).sum() / len(outputs)

    losses = 0
    for lh, gt in zip(outputs, labels):
        for ch in range(lh.shape[0]):
            if ch == 0: continue
            losses += topo_loss(lh[ch], gt == ch, topo_size=900) / (lh.shape[0] - 1)
    return losses / len(outputs)


def topo_loss(likelihood_tensor, gt_tensor, topo_size=900):
    """
    Calculate the topology loss of the predicted image and ground truth image
    Warning: To make sure the topology loss is able to back-propagation, likelihood
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.
    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The ground-truth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100
    Returns:
        loss_topo        :   The topology loss value (tensor)
    """
    device = likelihood_tensor.device
    lh = torch.sigmoid(likelihood_tensor).clone().squeeze().unsqueeze(0).cpu().detach().numpy()
    gt = gt_tensor.clone().squeeze().unsqueeze(0).cpu().detach().numpy()

    # dummy_lh, dummy_gt = np.ones_like(lh), np.ones_like(gt)
    # lh = np.concatenate([dummy_lh, lh, dummy_lh], axis=0)
    # gt = np.concatenate([dummy_gt, gt, dummy_gt], axis=0)

    crit_weight_map, crit_ref_map = np.zeros_like(lh), np.zeros_like(lh)

    for y in range(0, lh.shape[0], topo_size):
        for x in range(0, lh.shape[1], topo_size):

            # roi 떼기
            roi_lh = lh[y:min(y + topo_size, lh.shape[0]), x:min(x + topo_size, lh.shape[1])]
            roi_gt = gt[y:min(y + topo_size, gt.shape[0]), x:min(x + topo_size, gt.shape[1])]

            # 둘 중 하나라도 flat 하면 pass
            if np.min(roi_lh) == 1 or np.max(roi_lh) == 0: continue
            if np.min(roi_gt) == 1 or np.max(roi_gt) == 0: continue

            # Get the critical points of predictions and ground truth
            diag_lh, birth_lh, death_lh, bool_lh = get_cirit_pnts(roi_lh)
            diag_gt, birth_gt, death_gt, bool_gt = get_cirit_pnts(roi_gt)

            # If the pairs not exist, continue for the next loop
            if not (bool_lh and bool_gt): continue

            force_list, fix_hole_idx, rm_hole_idx = compute_dgm_force(diag_lh, diag_gt, pers_thresh=0.03)

            if len(fix_hole_idx) > 0 or len(rm_hole_idx) > 0:

                # Fix mode
                for idx in fix_hole_idx:
                    # Birth check -> push "birth" to Weight and Ref
                    if 0 <= birth_lh[idx][0] < lh.shape[0] and 0 <= birth_lh[idx][1] < lh.shape[1]:
                        crit_weight_map[y + birth_lh[idx][0], x + birth_lh[idx][1]] = 1
                        crit_ref_map[y + birth_lh[idx][0], x + birth_lh[idx][1]] = 0
                    # Death check -> push "death" to Weight and Ref
                    if 0 <= death_lh[idx][0] < lh.shape[0] and 0 <= death_lh[idx][1] < lh.shape[1]:
                        crit_weight_map[y + death_lh[idx][0], x + death_lh[idx][1]] = 1
                        crit_ref_map[y + death_lh[idx][0], x + death_lh[idx][1]] = 0

                # Remove mode
                for idx in rm_hole_idx:
                    # Birth check -> push "birth" to Weight
                    if 0 <= birth_lh[idx][0] < lh.shape[0] and 0 <= birth_lh[idx][1] < lh.shape[1]:
                        crit_weight_map[y + birth_lh[idx][0], x + birth_lh[idx][1]] = 1
                        # Death check in birth
                        if 0 <= death_lh[idx][0] < lh.shape[0] and 0 <= death_lh[idx][1] < lh.shape[1]:
                            # push "death" instead of birth to Ref
                            crit_ref_map[y + birth_lh[idx][0], x + birth_lh[idx][1]] = \
                                lh[death_lh[idx][0], death_lh[idx][1]]
                        else:
                            # Birth -> 1
                            crit_ref_map[y + birth_lh[idx][0], x + birth_lh[idx][1]] = 1
                    # Death check
                    if 0 <= death_lh[idx][0] < lh.shape[0] and 0 <= death_lh[idx][1] < lh.shape[1]:
                        crit_weight_map[y + death_lh[idx][0], x + death_lh[idx][1]] = 1
                        # Birth check in death
                        if 0 <= birth_lh[idx][0] < lh.shape[0] and 0 <= birth_lh[idx][1] < lh.shape[1]:
                            # push "birth" instead of death to Ref
                            crit_ref_map[y + death_lh[idx][0], x + death_lh[idx][1]] = \
                                lh[birth_lh[idx][0], birth_lh[idx][1]]
                        else:
                            # Death -> 0
                            crit_ref_map[y + death_lh[idx][0], x + death_lh[idx][1]] = 0

    crit_weight_map = torch.tensor(crit_weight_map, dtype=torch.float).to(device)
    crit_ref_map = torch.tensor(crit_ref_map, dtype=torch.float).to(device)
    # Measuring the MSE loss between predicted critical points and reference critical points
    loss = (((likelihood_tensor * crit_weight_map) - crit_ref_map) ** 2).sum()
    return loss


def get_cirit_pnts(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)
    Args:
        likelihood: Likelihood image from the output of the neural networks
    Returns:
        pd_lh:  persistence diagram.
        birth: Birth critical points.
        death: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.
    """
    lh = 1 - likelihood
    lh_flat = np.asarray(lh).flatten()
    lh_cubic = gd.CubicalComplex(dimensions=(lh.shape[0], lh.shape[1]), top_dimensional_cells=lh_flat)
    # lh_cubic = gd.SimplexTree(dimensions=(1, lh.shape[1]), top_dimensional_cells=lh_flat)

    diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()    # pairs_lh -> [N of points, 2]

    # If there is no pair, return False to skip
    if len(pairs_lh[0]) == 0: return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    persist_diagram, birth, death = [], [], []
    for pair in pairs_lh[0][0]:
        # pair = pairs_lh[0][0][i]
        persist_diagram.append([lh_flat[pair[0]], lh_flat[pair[1]]])
        birth.append([pair[0] // lh.shape[1], pair[0] % lh.shape[1]])
        death.append([pair[1] // lh.shape[1], pair[1] % lh.shape[1]])

    return np.array(persist_diagram), np.array(birth), np.array(death), True


def compute_dgm_force(diag_lh, diag_gt, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image
    Args:
        diag_lh: likelihood persistent diagram.
        diag_gt: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False
    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process
    """
    lh_pers = abs(diag_lh[:, 1] - diag_lh[:, 0])
    if diag_gt.shape[0] == 0:
        gt_pers, gt_num_holes = None, 0
    else:
        gt_pers = diag_gt[:, 1] - diag_gt[:, 0]
        gt_num_holes = gt_pers.size  # number of holes in gt

    if gt_pers is None or gt_num_holes == 0:
        idx_holes_to_fix = list()
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list()
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect]
        else:
            idx_holes_perfect = list()

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_num_holes]

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_num_holes:]

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(diag_lh.shape)

    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - diag_lh[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - diag_lh[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / math.sqrt(2.0)

    if do_return_perfect:
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove


def get_mse_loss(outputs, labels, power=None):
    if power:
        return torch.abs(torch.sum(outputs.view(-1) - labels.view(-1)) ** power)
    else:
        return F.mse_loss(outputs.view(-1), labels.view(-1))


def get_boundary_loss(outputs, soft_max=False):
    if soft_max:
        soft_max_outputs = torch.Tensor(outputs)
    else:
        soft_max_outputs = F.softmax(torch.Tensor(outputs), 1)
    # smoothed_sinus_qrs = F.conv1d(soft_max_outputs[:,1:,:], torch.ones((3,1,5)), padding='same', groups=3)
    smoothed_sinus_qrs = F.avg_pool1d(soft_max_outputs[:, 1:, :], 20, 10)
    loss = -torch.log((1 - torch.prod(smoothed_sinus_qrs[:, 0:2, :], dim=1)))
    # -torch.log((1 - torch.prod(smoothed_sinus_qrs[:, range(0, 4, 2), :], dim=1)))
    # -torch.log((1 - torch.prod(smoothed_sinus_qrs[:, 1:, :], dim=1)))
    return torch.max(loss)


# max_pooling base
def get_boundary_loss2(outputs, soft_max=False):
    if soft_max:
        soft_max_outputs = torch.Tensor(outputs)
    else:
        soft_max_outputs = F.softmax(torch.Tensor(outputs), 1)
    pooled_qrs = F.max_pool1d(soft_max_outputs, 20, 10)
    # smoothed_sinus_qrs = F.conv1d(soft_max_outputs[:,1:,:], torch.ones((3,1,5)), padding='same', groups=3)
    loss = -torch.log((1 - torch.prod(pooled_qrs[:, 0:2, :], dim=1))) * (1 - pooled_qrs[:, 0, :])
    -torch.log((1 - torch.prod(pooled_qrs[:, range(0, 4, 2), :], dim=1))) * (1 - pooled_qrs[:, 0, :])
    -torch.log((1 - torch.prod(pooled_qrs[:, 1:, :], dim=1))) * (1 - pooled_qrs[:, 0, :])
    return torch.max(loss)


def get_IOU(window1: tuple or list, window2: tuple or list):
    all_pnt = list(window1) + list(window2)
    all_pnt = sorted(all_pnt)

    union = all_pnt[-1] - all_pnt[0]
    intersection = all_pnt[2] - all_pnt[1]

    return intersection / union
