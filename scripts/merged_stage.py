import os
import time
from collections import defaultdict
import shutil
import wfdb.processing
import neurokit2 as nk
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts import stage_1, stage_2
from src.common.setup import Configure
from src.common.path import check_dir, path_finder
from src.fileio import DatasetParser_v2, get_holter_analysis
from src.data import data_preprocess, ECG_dataset, Collate_seq, recursive_peak_merge
from src.computer import R_Count, remove_tachy, idx_to_onoff
from src.verify import plot_seg, plot_compare
from AI_template.AI_common import operation
from AI_template.AI_common.measure import MarginalMetric, fbeta
from AI_template.AI_common.operation import print_confusion_matrix, label_to_onoff
from AI_template.AI_common.view import seg_plot

VIEW_SVT = True

def run(data_directory='', run_directory=''):
    # --------------------------------- Initialize --------------------------------- #
    print('\nRun for merged stage ...')
    # --------------------------------- Set stage 1 --------------------------------- #
    stg1_direc = os.path.join(run_directory, './stage_1')
    Configure.init_cfg()
    cfg1 = Configure.get_cfg(os.path.join(stg1_direc, '_checkpoints/config_stage_1.json'))

    # --------------------------------- Load data @ stage 1 --------------------------------- #
    parser = DatasetParser(data_directory)
    if not os.path.isdir(data_directory):
        inference_dataset = ECG_dataset(**parser.read_db(data_directory, sec=3600 * 12))
    else:
        inference_dataset = ECG_dataset(**parser.parse(mode='test'))
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False,
                                  collate_fn=Collate_seq(sequence=['split']).wrap())

    # --------------------------------- Inference & Postprocess @ stage 1 --------------------------------- #
    prediction_stg1 = stage_1.core(mode='test', loader_dict={'test': inference_loader},
                                   model_save_path=path_finder(stg1_direc, '.pt')[0])
    # postprocessing
    cursor, fin_dict = 0, defaultdict(list)
    for col in inference_loader.__iter__():
        n = col['x'].__len__()
        fin_dict['x'].append(col['raw'])
        fin_dict['sec10'].append(col['x'].squeeze())
        fin_dict['raw_y'].append(col['y'])
        pred_onoff = operation.label_to_onoff(prediction_stg1[cursor:cursor + n], sense=5)
        pred_onoff = [operation.sanity_check(oo) for oo in pred_onoff]
        pred_onoff = recursive_peak_merge(pred_onoff, len(pred_onoff) // 2, 250, 2)[0]
        fin_dict['onoff'].append(pred_onoff[pred_onoff.max(1) < len(col['raw'])])
        fin_dict['y'].append(operation.onoff_to_label(np.array(pred_onoff), length=len(col['raw'])).tolist())
        fin_dict['fn'].append(col['fn'])
        cursor += n

    # --------------------------------- Load data @ stage 2 --------------------------------- #
    stg2_direc = os.path.join(run_directory, './stage_2')
    Configure.init_cfg()
    cfg2 = Configure.get_cfg(os.path.join(stg2_direc, '_checkpoints/config_stage_2.json'))
    inference_dataset2 = ECG_dataset(x=np.stack(fin_dict['x']), y=np.stack(fin_dict['y']),
                                     fn=fin_dict['fn'], onoff=fin_dict['onoff'])
    inference_loader2 = DataLoader(inference_dataset2, batch_size=1, shuffle=False,
                                   collate_fn=Collate_seq(sequence=['r_reg'], zit=False, resamp=cfg2.model.dim_in).wrap())

    # --------------------------------- Inference & Postprocess @ stage 2 --------------------------------- #
    prediction_stg2 = stage_2.core(mode='test', loader_dict={'test': inference_loader2},
                                   model_save_path=path_finder(stg2_direc, '.pt')[0])
    # postprocessing
    cursor = 0
    for i, onoff in enumerate(inference_dataset2.onoff):
        n = onoff.__len__()
        r_loc = prediction_stg2[cursor:cursor+n]
        r_idx = onoff[:, 0] + (onoff[:, 1] - onoff[:, 0]) * r_loc
        fin_onoff = np.concatenate((onoff, r_idx.reshape(-1, 1)), axis=1).astype(int)
        fin_dict['fin_onoff'].append(fin_onoff[fin_onoff[:, -1] < len(fin_dict['x'][i])])
        cursor += n

    # --------------------------------- Conf Mat --------------------------------- #
    bi_mat = np.zeros((cfg1.model.ch_out, cfg1.model.ch_out))
    iii = 0
    for c, o, l in zip(np.concatenate(fin_dict['sec10']), prediction_stg1, np.concatenate(fin_dict['raw_y'])):
        iii += 1
        bi_mat += MarginalMetric(prediction=o, label=l, num_class=cfg1.model.ch_out).marginal_mat

        if VIEW_SVT:
            oo = np.array(label_to_onoff(l))[:, -1]
            conv1 = np.convolve(oo, [2, 2, 2])
            conv2 = np.convolve(oo, [1, 2, 3])
            if sum([c1 == c2 == 12 for c1, c2 in zip(conv1, conv2)]):
                plot_compare(os.path.join('./svt_check'), c, l, o, fn=str(iii))

    print_confusion_matrix(bi_mat, class_name=['B', 'N', 'A', 'V'])
    print(f'\n\tBeat Accuracy: {100 * bi_mat[1:, 1:].sum() / bi_mat.sum():.2f} %')
    print(f'\tF1-score: {100 * fbeta(bi_mat[1:, 1:], beta=1, num_classes=cfg1.model.ch_out - 1):.2f} %')

    # --------------------------------- R-count --------------------------------- #
    rpeaks = inference_dataset.rp
    r_counter = R_Count(tol=9)
    print('\nMODEL')
    for i, fn in enumerate(fin_dict['fn']):
        r_counter.cal(rpeaks[i], fin_dict['fin_onoff'][i][:, -1])
    r_counter.logging()
    print('\nNEUROKIT')
    r_counter.clear()
    for i, raw in enumerate(fin_dict['x']):
        signal, info = nk.ecg_peaks(raw, sampling_rate=250, method="neurokit")
        r_counter.cal(rpeaks[i], info['ECG_R_Peaks'])
    r_counter.logging()

    return fin_dict


def test4license(save_direc, run_dir_1='./stage_1', run_dir_2='./stage_2'):
    map_dict = {1: ('N', ""), 2: ('S', ""), 3: ('V', ""), 98: ("+", "(AFIB"), 99: ("+", "(N")}

    check_dir(save_direc)
    parser = DatasetParser(data_directory='', BIN='license', pad=True)
    for db_name in parser.db_path.keys():
        check_dir(os.path.join(save_direc, db_name))

        for idx in range(len(parser.db_filenames[db_name])):
            serial, raw_signal, filtered_signal, test_x, rpeaks = parser.parse(db=db_name, idx=idx)
            if raw_signal is None:
                print('\nCannot find %s... Pass!' % serial)
                continue
            print('\nInference for %s in %s...' % (serial, db_name))
            test_dataset = ECG_dataset(test_x, fn=[serial]*len(test_x), augmentation=False)
            onoff_matrix = run(data_directory='', run_dir_1=run_dir_1, run_dir_2=run_dir_2, testset=test_dataset)
            if db_name == 'aha':
                trimmed_mat = onoff_matrix[300 * 250 < onoff_matrix[:, -2]]   # cut first 300 seconds.
                trimmed_mat = trimmed_mat[trimmed_mat[:, -2] < len(raw_signal)]     # cut dummy signal
            else:
                trimmed_mat = onoff_matrix[onoff_matrix[:, -2] < len(raw_signal)]   # cut dummy signal

            # # check tachy - find beats faster than 180 heart rate (333ms ).
            # ref_hr = 180
            # rr_interval = np.diff([0] + trimmed_mat[:, -2].tolist())
            # bool_tachy = rr_interval < (60/ref_hr) / 0.004
            # tachy_onoff = label_to_onoff(bool_tachy, sense=3)
            # for on, off, _ in tachy_onoff:
            #     for i_tachy in range(on, off+2):
            #         if trimmed_mat[i_tachy, -1] == 1:   trimmed_mat[i_tachy, -1] = 2

            pred_cls = trimmed_mat[:, -1] == 3
            sensing = label_to_onoff(pred_cls, sense=5)
            print('\t Find %d long v run' % len(sensing))
            for sens in sensing:
                print('\t\t run %d' % (sens[1] - sens[0] + 1))

            if os.path.exists(parser.db_filenames[db_name][idx] + '.txt'):
                fib_idx = np.loadtxt(parser.db_filenames[db_name][idx] + '.txt', dtype=np.int32)
                fib_mat = []
                for onoff in idx_to_onoff(fib_idx, sense=3):
                    on, off = onoff[0] * 2500, (onoff[1] + 1) * 2500
                    fib_mat.append([on] * 3 + [98]), fib_mat.append([off] * 3 + [99])
                    condition = (on < trimmed_mat[:, -2]) * (trimmed_mat[:, -2] < off) * trimmed_mat[:, -1] == 2
                    trimmed_mat[condition, -1] = 1
                if fib_mat: trimmed_mat = np.concatenate((trimmed_mat, fib_mat), axis=0)

            # resample & sort
            trimmed_mat[:, -2] = trimmed_mat[:, -2] * parser.db_fs[db_name] / 250
            trimmed_mat = np.array(sorted(trimmed_mat, key=lambda x: x[-2]))
            r_indices, sym_aux = trimmed_mat[:, -2], np.zeros((len(trimmed_mat), 2), dtype=object)
            for k, v in map_dict.items():
                sym_aux[trimmed_mat[:, -1] == k] = v
            # # aux = wfdb.processing.compute_hr(len(raw_signal), r_indices, parser.db_fs[db_name])
            # aux = 60 / wfdb.processing.calc_rr(r_indices, fs=250, rr_units='seconds')
            # aux = [''] + list(np.array(np.array(aux, dtype=np.longfloat), dtype=np.str))
            wfdb.wrann(serial, 'hui', sample=r_indices, symbol=sym_aux[:, 0], aux_note=sym_aux[:, 1],
                       fs=parser.db_fs[db_name], write_dir=os.path.join(save_direc, db_name))


def pre_anno(Diag, data_directory='./_data/pre_anno', run_dir_1='./stage_1', run_dir_2='./stage_2', r_only=False):
    data_directory = os.path.join(data_directory, Diag)
    if Diag == 'af_data_for_annotation':
        workers = ['성훈', '준호', '광로']
    else:
        workers = ['재성', '진국', '성재', '명규', '상일', '권우', '성진', '지우', '원선', '상규']

    # --------------------------------------------------------------------------- #
    # ---------------------------       Pre-set       --------------------------- #
    # --------------------------------------------------------------------------- #

    # --------------------------- Set directory --------------------------- #
    checks_direc_1 = os.path.join(run_dir_1, '_checkpoints')
    output_direc_1 = os.path.join(run_dir_1, '_outputs')
    check_dir(run_dir_1, checks_direc_1, output_direc_1)
    cfg1 = Configure.get_cfg(os.path.join(checks_direc_1, 'config_stage_1.json'))

    checks_direc_2 = os.path.join(run_dir_2, '_checkpoints')
    output_direc_2 = os.path.join(run_dir_2, '_outputs')
    check_dir(run_dir_2, checks_direc_2, output_direc_2)
    cfg2 = Configure.get_cfg(os.path.join(checks_direc_2, 'config_stage_2.json'))

    device = torch.device(f'cuda:%d' % device_number if torch.cuda.is_available() else 'cpu')
    cfg1.run.device, cfg2.run.device = device, device
    cfg1.run.class_weight = torch.Tensor(cfg1.run.class_weight).to(device)
    if cfg1.data.scaler != 'minmax':   cfg1.data.scaler = 200

    model_save_path_1 = os.path.join(checks_direc_1, 'ECG_seg_checkpoint_QRS.pt')

    output_name_2 = 'rp_regression_checkpoint_w{}_k{}_d{}_o{}s{}_ro{}_ed{}_aug{}_resamp{}.pt'.format(
        cfg2.model.width,
        cfg2.model.kernel_size,
        cfg2.model.depth,
        cfg2.model.order,
        cfg2.model.stride,
        cfg2.model.regression_order,
        cfg2.model.embedding_dims,
        cfg2.data.augmentation,
        cfg2.model.resample_len
    )
    model_save_path_2 = os.path.join(checks_direc_2, output_name_2)

    # --------------------------- Load data --------------------------- #
    # x: input / y: target / fn: filenames / rp: r-peak
    parser = DatasetParser(data_directory, cfg=cfg1, BIN='txt')
    pre_fn, raw, filtered, pre_x, _ = parser.parse()
    pre_dataset = ECG_dataset(pre_x, np.zeros_like(pre_x), np.zeros(len(pre_x)), mode='test', augmentation=False)
    if cfg1.run.combine_categories:
        pre_dataset.toggle(['test'] + cfg1.run.combine_categories)
    pre_loader = torch.utils.data.DataLoader(pre_dataset, batch_size=cfg1.run.batch_size_eval, shuffle=False)

    # --------------------------- Modeling & Scheduling --------------------------- #
    model1 = stage_1.get_model(**cfg1.model.__dict__)
    model2 = stage_2.get_model(**cfg2.model.__dict__)
    model1.to(cfg1.run.device)
    model2.to(cfg2.run.device)

    # --------------------------------------------------------------------------- #
    # ---------------------------        Test         --------------------------- #
    # --------------------------------------------------------------------------- #
    print('\nTest model 1')
    model1.load_state_dict(torch.load(model_save_path_1, map_location=torch.device("cpu")), strict=False)
    _, _, _, test_out = stage_1.test(cfg=cfg1, model=model1, loader=pre_loader, metric=False)
    if not cfg1.model.output_smoothing:
        print('\nPostprocessing...')
        _, test_out = stage_1.post(cfg1, test_out)

    test_data2 = []
    scaled_pre_x, _, _ = data_preprocess(pre_x.squeeze(), mode='t', seed=SEED, scaling=True, scaler=cfg2.data.scaler)

    # test에서는 target_rpead_index가 필요없으니 dummy value로 채워둡니다.
    for qrs_annotation in test_out:  # Fix : morphed -> cleared
        try:
            if sum(qrs_annotation) > 1:
                test_data2.append(
                    np.insert(sanity_check(label_to_onoff(qrs_annotation), incomplete_only=True), 2, 0, axis=1))
            else:
                test_data2.append(np.array([]))
        except np.AxisError:
            # 특정 10초 데이터에서 QRS가 검출되지 않앗을 경우, 빈 array에 값을 억지로 끼워넣다 np.AxisError 발생
            # 일단 dummy data를 삽입 후 진행
            test_data2.append(np.array([[0, 0, 0, outside_idx]]))

    test_qrs_rpeak_input, test_qrs_rpeak_target, _, test_qrs_class_target, _ = \
        normalizing_rpeak_index(scaled_pre_x, test_data2, cfg2.model.resample_len,
                                additional_classification=cfg2.run.additional_classification)

    # --------------------------------- Dataset & Loader --------------------------------- #
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_qrs_rpeak_input),
                                                  torch.Tensor(test_qrs_rpeak_target),
                                                  torch.Tensor(test_qrs_class_target))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg2.run.batch_size_eval, shuffle=False)

    print('\nTest model 2')
    model2.load_state_dict(torch.load(model_save_path_2, map_location=torch.device("cpu")), strict=True)
    _, _, test_out_reg = stage_2.test(cfg=cfg2, model=model2, loader=test_loader, return_output=True)

    if filtered is None:
        # 태생이 short data
        raw = filtered = pre_x.reshape(-1) / 200
        cursor = 0
        test_out2 = np.copy(test_data2)

        for i in range(len(test_out2)):
            unit_onoff = test_out2[i]
            unit_reg = test_out_reg[cursor:cursor + len(unit_onoff)].squeeze()
            unit_reg = unit_reg * (unit_onoff[:, 1] - unit_onoff[:, 0]) + unit_onoff[:, 0]
            unit_onoff[:, -2] += np.array(unit_reg, dtype=np.int32) + 2500 * i
            unit_onoff[:, :-2] += 2500 * i
            cursor += len(unit_onoff)
            test_out2[i] = unit_onoff

        term = len(test_out2) // len(pre_fn)
        test_fin = []
        for ii in range(len(pre_fn)):
            chunk = test_out2[ii*term:(ii+1)*term]
            # chunk = recursive_peak_merge(chunk % 2500, len(chunk) // 2, 250, 2)
            # chunk = unit_to_full(chunk.squeeze())
            chunk = np.concatenate(chunk, axis=0)
            chop_removed = remove_tachy(chunk[:, -2])
            chop_removed_on_off = np.array([oo for oo in chunk if oo[-2] in chop_removed])
            chop_removed_on_off = chop_removed_on_off[chop_removed_on_off[:, -1] != outside_idx]
            test_fin.append(chop_removed_on_off)

        result_on_off = np.concatenate(test_out2)
        # result_on_off = result_on_off[result_on_off[:, -1] != outside_idx]
        # result_rpeak = test_out_regression.squeeze() * (result_on_off[:, 1] - result_on_off[:, 0])
        # result_rpeak += result_on_off[:, 0]
        # result_on_off[:, -2] += np.array(result_rpeak, dtype=np.int32)  # dummy value에 prediction 채워넣기
        # chop_removed_on_off = result_on_off
    else:
        # 태생이 long data
        term = 1
        result_on_off = np.concatenate(test_data2)
        result_on_off = result_on_off[result_on_off[:, -1] != outside_idx]
        result_rpeak = test_out_reg.squeeze() * (result_on_off[:, 1] - result_on_off[:, 0]) + result_on_off[:, 0]
        result_on_off[:, -2] = np.array(result_rpeak, dtype=np.int32)  # dummy value에 prediction 채워넣기
        chop_removed = remove_tachy(result_on_off[:, -2])
        chop_removed_on_off = np.array([oo for oo in result_on_off if oo[-2] in chop_removed])
        test_fin = [chop_removed_on_off]

    # plot_dict = {'signal': raw, 'filtered': filtered, 'result': chop_removed_on_off[:, -2:]}
    # comparison = plotter.Comparison()
    # comparison.make_gui(plot_dict)

    # --------------------------------- Annotation --------------------------------- #
    # W_onoff
    seg_to_anno = [stage_1.Dx_naming(
        onoff_to_label(sanity_check(label_to_onoff(c), incomplete_only=True)).squeeze()) for c in test_out]
    # seg_to_anno = [stage_1.Dx_naming(test_out[i*term:(i+1)*term].reshape(-1)) for i in range(len(test_out)//term)]
    rp_to_anno = [[str(o % (2500*term)) + ', R' for o in oo[:, -2]] for oo in test_fin]

    fin = []
    for seg, rp in zip(seg_to_anno, rp_to_anno):
        fin.append(sorted(seg + rp, key=lambda x: int(x.split(',')[0])))

    if r_only:    fin = rp_to_anno

    # Save as txt format
    now_day = time.strftime('%Y%m%d')
    save_path = data_directory.replace(Diag, now_day)
    if not os.path.isdir(save_path):        os.mkdir(save_path)
    for worker in workers:
        if not os.path.isdir(os.path.join(save_path, worker)):
            os.mkdir(os.path.join(save_path, worker))
        check_dir(os.path.join(save_path, worker, 'data_' + Diag))
        check_dir(os.path.join(save_path, worker, 'preanno_' + Diag))

    seeed = np.random.randint(0, 9999)
    print(seeed)
    np.random.seed(seeed)
    roulette = np.array_split(np.random.permutation(np.arange(len(fin))), len(workers))
    worker_roulette = np.random.permutation(workers)
    print(worker_roulette[:len(fin)])

    for i_work, name in enumerate(worker_roulette):
        idx = roulette[i_work]
        for i in idx:
            shutil.copy(os.path.join(data_directory, pre_fn[i]), os.path.join(save_path, name, 'data_'+Diag))
            if term == 1:
                with open(os.path.join(save_path, name, 'preanno_'+Diag, pre_fn[i].replace('.txt', '')+'-annotations.txt'), 'w+', encoding='utf-8') as lf:
                    lf.write('\n'.join(fin[i]))
            else:
                fin_ = [[int(f.split(',')[0]) for f in fi] for fi in fin]
                np.save(os.path.join(save_path, name, 'preanno_' + Diag,
                                       pre_fn[i].replace('.txt', '') + '_annotation'), fin_[i])


if __name__ == '__main__':
    print('annotation test')
    path = 'E:/database/openDB'
    anno_path = os.path.join(path, 'ahadb')
    pred_path = os.path.join('../for_license', 'aha')
    pat_id = '1201'

    import wfdb

    anno = wfdb.rdann(os.path.join(anno_path, pat_id), extension='atr')
    pred = wfdb.rdann(os.path.join(pred_path, pat_id), extension='hui')

    e, d = 0, 0
    for m_anno, m_pred in zip(anno.__dict__.items(), pred.__dict__.items()):
        same = False
        k_anno, k_pred = m_anno[0], m_pred[0]
        v_anno, v_pred = m_anno[1], m_pred[1]
        if '__len__' in v_anno.__dir__() and '__len__' in v_pred.__dir__():
            if v_anno.__len__() == v_pred.__len__():
                same = True
        else:
            if v_anno == v_pred:
                same = True

        if same:
            e += 1
            print('Equal')
        else:
            d += 1
            print('Diff')

    print('\nEqual:%d\tDiff:%d' % (e, d))

    rr_anno = wfdb.ann2rr(os.path.join(anno_path, pat_id), extension='atr')
    rr_pred = wfdb.ann2rr(os.path.join(pred_path, pat_id), extension='hui')

    print('Done')

