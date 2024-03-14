import pandas as pd
import numpy as np
import wfdb
import os
import ecg_gudb_database

from src.computer import remove_tachy


def DB_to_csv(directory, DB_name, save_path='./_data/openDB'):
    dfs_ecg, dfs_rpeaks = [], []

    if 'mit' in DB_name:
        if 'normal' in DB_name:
            print('MIT-BIH normal dataset')
            root = "mit-bih-normal-sinus-rhythm-database-1.0.0/"
            filenames = []
            for f in os.listdir(os.path.join(directory, root)):
                if '.dat' in f:
                    filenames.append(os.path.join(directory, root, f))

            cursor = 0
            for participant, file in enumerate(filenames):
                print("Participant: " + str(participant + 1) + "/" + str(len(filenames)))

                # Get signal
                info = wfdb.rdrecord(file[:-4], sampfrom=0, sampto=5000)
                if 'ECG2' not in info.sig_name:
                    continue
                lead2_idx = info.sig_name.index('ECG2')
                data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, lead2_idx]})
                data["Participant"] = "MIT-Normal_%.2i" % participant
                data["Sample"] = range(len(data))
                data["Sampling_Rate"] = 128
                data["Database"] = "MIT-Normal"

                # getting annotations
                anno = wfdb.rdann(file[:-4], 'atr')
                anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
                anno = pd.DataFrame({"Rpeaks": anno})
                anno["Participant"] = "MIT-Normal_%.2i" % participant
                anno["Sampling_Rate"] = 128
                anno["Database"] = "MIT-Normal"

                # Select only 12h of recording (otherwise it's too big)
                data = data[:460800 * 12].reset_index(drop=True)
                anno = anno[(anno["Rpeaks"] > 0) & (anno["Rpeaks"] <= 460800 * 12)].reset_index(drop=True)
                anno["Rpeaks"] = anno["Rpeaks"]
                anno["Rpeaks"] += cursor
                # Store with the rest
                dfs_ecg.append(data)
                dfs_rpeaks.append(anno)
                cursor += len(data)

        elif 'arrhythmia' in DB_name:
            print('MIT-BIH arrhythmia dataset')

            def read_file(file, participant):
                """Utility function
                """
                # Get signal
                info = wfdb.rdrecord(file[:-4], sampfrom=0, sampto=5000)
                if 'MLII' not in info.sig_name:
                    return None, None
                lead2_idx = info.sig_name.index('MLII')
                data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, lead2_idx]})
                data["Participant"] = "MIT-Arrhythmia_%.2i" % participant
                data["Sample"] = range(len(data))
                data["Sampling_Rate"] = 360
                data["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

                # getting annotations
                anno = wfdb.rdann(file[:-4], 'atr')
                anno = np.unique(anno.sample[np.in1d(anno.symbol,
                                                     ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j',
                                                      'n', 'E', '/', 'f', 'Q', '?', '!'])])
                anno = pd.DataFrame({"Rpeaks": anno})
                anno["Participant"] = "MIT-Arrhythmia_%.2i" % participant
                anno["Sampling_Rate"] = 360
                anno["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

                return data, anno

            root = "mit-bih-arrhythmia-database-1.0.0/"
            filenames = []
            for f in os.listdir(os.path.join(directory, root)):
                if '.dat' in f:
                    filenames.append(os.path.join(directory, root, f))

            offset = 0
            for participant, file in enumerate(filenames):
                print("Participant: " + str(participant + 1) + "/" + str(len(filenames)))

                data, anno = read_file(file, participant)

                if data is None:
                    continue

                anno["Rpeaks"] += offset
                offset += len(data)

                # Store with the rest
                dfs_ecg.append(data)
                dfs_rpeaks.append(anno)

                # # Store additional recording if available
                # if "x_" + os.path.split(file)[1] in os.listdir(os.path.join(directory, root, 'x_mitdb')):
                #     print("  - Additional recording detected.")
                #     data, anno = read_file(os.path.join(os.path.split(file)[0], 'x_mitdb',
                #                                         "x_" + os.path.split(file)[1]), participant)
                #     anno["Rpeaks"] += offset
                #     offset += len(data)
                #     # Store with the rest
                #     dfs_ecg.append(data)
                #     dfs_rpeaks.append(anno)

        elif 'noise' in DB_name:
            print('MIT-BIH noise stress dataset')

            def read_file(file, participant):
                """Utility function
                """
                # Get signal
                info = wfdb.rdrecord(file[:-4], sampfrom=0, sampto=5000)
                if 'MLII' not in info.sig_name:
                    return None, None
                lead2_idx = info.sig_name.index('MLII')
                data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, lead2_idx]})
                data["Participant"] = "MIT-Noise_%.2i" % participant
                data["Sample"] = range(len(data))
                data["Sampling_Rate"] = 360
                data["Database"] = "MIT-Noise"

                # getting annotations
                anno = wfdb.rdann(file[:-4], 'atr')
                anno = np.unique(anno.sample[np.in1d(anno.symbol,
                                                     ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j',
                                                      'n', 'E', '/', 'f', 'Q', '?', '!'])])
                anno = pd.DataFrame({"Rpeaks": anno})
                anno["Participant"] = "MIT-Noise_%.2i" % participant
                anno["Sampling_Rate"] = 360
                anno["Database"] = "MIT-Noise"

                return data, anno

            root = "mit-bih-noise-stress-test-database-1.0.0/"
            filenames = []
            for f in os.listdir(os.path.join(directory, root)):
                if '.dat' in f:
                    if f[:-4] not in ['bw', 'em', 'ma']:
                        snr = f[-6:-4]
                        if '_' not in snr:
                            if int(snr) >= 18:
                                filenames.append(os.path.join(directory, root, f))

            offset = 0
            for participant, file in enumerate(filenames):
                print("Participant: " + str(participant + 1) + "/" + str(len(filenames)))

                data, anno = read_file(file, participant)
                if data is None:
                    continue

                anno["Rpeaks"] += offset
                offset += len(data)

                # Store with the rest
                dfs_ecg.append(data)
                dfs_rpeaks.append(anno)

    elif 'gudb' in DB_name:
        print('GUDB dataset')
        offset = 0
        for participant in range(25):
            print("Participant: " + str(participant + 1) + "/25")
            for i, experiment in enumerate(ecg_gudb_database.GUDb.experiments):
                print("  - Condition " + str(i + 1) + "/5")
                # creating class which loads the experiment
                ecg_class = ecg_gudb_database.GUDb(participant, experiment)

                # Chest Strap Data - only download if R-peaks annotations are available
                if ecg_class.anno_cables_exists:
                    data = pd.DataFrame({"ECG": ecg_class.einthoven_II * 1000})
                    data["Participant"] = "GUDB_%.2i" % participant
                    data["Sample"] = range(len(data))
                    data["Sampling_Rate"] = 250
                    data["Database"] = "GUDB_" + experiment

                    # getting annotations
                    anno = pd.DataFrame({"Rpeaks": ecg_class.anno_cables})
                    anno["Participant"] = "GUDB_%.2i" % participant
                    anno["Sampling_Rate"] = 250
                    anno["Database"] = "GUDB (" + experiment + ")"

                    # Store with the rest
                    anno["Rpeaks"] += offset
                    offset += len(data)
                    dfs_ecg.append(data)
                    dfs_rpeaks.append(anno)

    elif 'fantasia' in DB_name:
        print('Fantasia dataset')
        root = "fantasia-database-1.0.0"
        filenames = []
        for f in os.listdir(os.path.join(directory, root)):
            if '.dat' in f:
                filenames.append(os.path.join(directory, root, f))

        for i, participant in enumerate(filenames):
            participant = participant.replace('.dat', '')
            data, info = wfdb.rdsamp(participant)

            # Get signal
            data = pd.DataFrame(data, columns=info["sig_name"])
            data = data[["ECG"]]
            data["Participant"] = "Fantasia_" + participant
            data["Sample"] = range(len(data))
            data["Sampling_Rate"] = info['fs']
            data["Database"] = "Fantasia"

            # Get annotations
            anno = wfdb.rdann(participant, 'ecg')
            anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
            anno = pd.DataFrame({"Rpeaks": anno})
            anno["Participant"] = "Fantasia_" + participant
            anno["Sampling_Rate"] = info['fs']
            anno["Database"] = "Fantasia"

            # Store with the rest
            dfs_ecg.append(data)
            dfs_rpeaks.append(anno)

    elif 'ludb' in DB_name:
        print('LUDB dataset')
        root = "lobachevsky-university-electrocardiography-database-1.0.1/data"

        # files = os.listdir("./ludb-1.0.0.physionet.org/")
        for participant in range(200):
            filename = str(participant + 1)

            data, info = wfdb.rdsamp(os.path.join(directory, root, filename))

            # Get signal
            data = pd.DataFrame(data, columns=info["sig_name"])
            data = data[["i"]].rename(columns={"i": "ECG"})
            data["Participant"] = "LUDB_%.2i" % (participant + 1)
            data["Sample"] = range(len(data))
            data["Sampling_Rate"] = info['fs']
            data["Database"] = "LUDB"

            # Get annotations
            anno = wfdb.rdann(os.path.join(directory, root, filename), 'atr_i')
            anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
            anno = pd.DataFrame({"Rpeaks": anno})
            anno["Participant"] = "LUDB_%.2i" % (participant + 1)
            anno["Sampling_Rate"] = info['fs']
            anno["Database"] = "LUDB"

            # Store with the rest
            dfs_ecg.append(data)
            dfs_rpeaks.append(anno)

    elif 'huinno' in DB_name.lower():
        print('HUINNO Patch dataset')
        root = "huinno"
        dirs = os.listdir(os.path.join(directory, root))
        participant, cursor, buff = 0, 0, 0
        for d in dirs:
            full_path = os.path.join(directory, root, d)
            files = os.listdir(os.path.join(full_path))
            for f in files:
                if os.path.splitext(f)[-1] == '.txt':
                    print("\tLoading: %s" % f)
                    signal = np.fromstring(open(os.path.join(full_path, f), 'r').read(), dtype=float, sep='\n')
                    data = pd.DataFrame({"ECG": signal})
                    data["Participant"] = "Patch_afib_%.2i" % (participant//2 + 1)
                    data["Sample"] = range(len(data))
                    data["Sampling_Rate"] = 250
                    data["Database"] = "PATCH"
                    participant += 1
                    buff = len(data)
                    dfs_ecg.append(data)
                elif os.path.splitext(f)[-1] == '.npy':
                    print("\tLoading: %s - annotation" % f)
                    peaks = np.load(os.path.join(full_path, f))
                    peaks = remove_tachy(np.array(sorted(set(peaks))))
                    anno = pd.DataFrame({"Rpeaks": peaks + cursor})
                    anno["Participant"] = "Patch_afib_%.2i" % (participant//2 + 1)
                    anno["Sampling_Rate"] = 250
                    anno["Database"] = "PATCH"
                    participant += 1
                    cursor += buff
                    dfs_rpeaks.append(anno)

    else:
        raise ValueError

    # Save
    if not os.path.isdir(os.path.join('./_data/openDB', DB_name)):
        os.mkdir(os.path.join('./_data/openDB', DB_name))
    pd.concat(dfs_ecg).to_csv(os.path.join(save_path, DB_name, 'ECGs.csv'), index=False)
    pd.concat(dfs_rpeaks).to_csv(os.path.join(save_path, DB_name, 'Rpeaks.csv'), index=False)


if __name__ == '__main__':
    DB_to_csv(directory='E:/database/openDB', DB_name='mit-normal')
    DB_to_csv(directory='E:/database/openDB', DB_name='mit-arrhythmia')
    DB_to_csv(directory='E:/database/openDB', DB_name='gudb')
    # DB_to_csv(directory='E:/database/openDB', DB_name='ludb')
    DB_to_csv(directory='E:/database/openDB', DB_name='fantasia')

