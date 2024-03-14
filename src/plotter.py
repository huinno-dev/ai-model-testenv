import numpy as np
import time
import sys
import tkinter as tk
import neurokit2 as nk
import threading
import platform


def runner(obj):
    total_lines = len(obj.data) // (obj.fs * obj.sec_in_line)
    while obj.lines_load + 3 < total_lines:
        obj.lines_add += 10
        obj.draw_graph()


class Comparison:
    canvas_left = None
    canvas_right = None
    canvas_info = None
    canvas_line = None
    canvas_width = 800
    sec_in_line = 30
    lines_load = 0
    lines_add = 100
    data = []
    filtered = []
    fs = 250
    vs_filtered = False

    sb = None

    px_in_1sec = canvas_width / sec_in_line
    px_in_1mv = px_in_1sec / 5 * 2

    r_peaks_left = []
    r_peaks_right = []
    qrs = []
    qrs_on = []
    qrs_off = []
    qrs_type = []

    def make_gui(self, dictionary=None):
        window = tk.Tk()
        window.title('Viewer')
        w = 1800
        h = 900
        ws = window.winfo_screenwidth()
        hs = window.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        window.geometry('%dx%d+%d+%d' % (w, h, x, y))

        st_line = tk.Label(window, text='Line')
        st_line.grid(row=0, column=0, sticky='w')
        st_left = tk.Label(window, text='filtered data')
        st_left.grid(row=0, column=1, sticky='w')
        st_right = tk.Label(window, text='raw data')
        st_right.grid(row=0, column=2, sticky='w')

        self.canvas_line = tk.Canvas(window, bg='white', width=50, height=800)
        self.canvas_line.grid(row=1, column=0)

        fr_left = tk.Frame(window)
        fr_left.grid(row=1, column=1)
        self.canvas_left = tk.Canvas(fr_left, bg='white', width=self.canvas_width, height=800)
        self.canvas_left.pack(side='left')

        fr_right = tk.Frame(window)
        fr_right.grid(row=1, column=2)
        self.canvas_right = tk.Canvas(fr_right, bg='white', width=self.canvas_width, height=800)
        self.canvas_right.pack(side='left')

        self.sb = tk.Scrollbar(fr_left, command=self.drag_scroll)
        self.sb.pack(side='right', fill='y')

        self.canvas_left.configure(yscrollcommand=self.sb.set)
        self.canvas_right.configure(yscrollcommand=self.sb.set)

        self.canvas_info = tk.Canvas(window, bg='white', width=100, height=800)
        self.canvas_info.grid(row=1, column=3)

        self.read_data(dictionary, fs=250)
        # self.read_data('../WFDB/raw_data_A006S.txt', fs=250)
        # self.read_data('patch_test_data/2021-05-11_12-25-11_A0002_raw_data.txt', fs=250)

        self.canvas_left.configure(scrollregion=(0, 0, self.canvas_width,
                                                 self.px_in_1mv * 3 * self.lines_load))
        self.canvas_left.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas_right.configure(scrollregion=(0, 0, self.canvas_width,
                                                  self.px_in_1mv * 3 * self.lines_load))
        self.canvas_right.bind('<MouseWheel>', self.on_mouse_wheel)

        self.canvas_info.configure(scrollregion=(0, 0, self.canvas_width,
                                                 self.px_in_1mv * 3 * self.lines_load))
        self.canvas_line.configure(scrollregion=(0, 0, self.canvas_width,
                                                 self.px_in_1mv * 3 * self.lines_load))
        # self.canvas_left.bind('<Motion>', self.on_mouse_hover)
        window.mainloop()

    def on_mouse_wheel(self, event):
        # https://stackoverflow.com/questions/17355902/tkinter-binding-mousewheel-to-scrollbar
        scroll = int(-1 * event.delta) if platform.system() == 'Darwin' else int(-1 * event.delta / 120)
        self.canvas_left.yview_scroll(scroll, "units")
        self.canvas_right.yview_scroll(scroll, "units")
        self.canvas_info.yview_scroll(scroll, "units")
        self.canvas_line.yview_scroll(scroll, "units")

    def drag_scroll(self, *args):
        self.canvas_left.yview(*args)
        self.canvas_right.yview(*args)
        self.canvas_info.yview(*args)
        self.canvas_line.yview(*args)

    def read_data(self, dictionary=None, fs=250):
        if dictionary is not None:
            self.data = dictionary['signal']
            self.filtered = dictionary['filtered']
            if isinstance(dictionary['result'], tuple):
                self.r_peaks_left = dictionary['result'][0][:, 0]
                self.r_peaks_right = dictionary['result'][1][:, 0]
                self.qrs_type = dictionary['result'][1][:, 1]
            else:
                self.r_peaks_right = dictionary['result'][:, 0]
                if self.vs_filtered:
                    self.r_peaks_left = dictionary['result'][:, 0]
                else:
                    self.detector_for_left()
                self.qrs_type = dictionary['result'][:, 1]

            t = threading.Thread(target=runner, args=(self, ))
            t.start()

    def draw_30s_graph(self, line):
        line_data = self.data[line * self.fs * self.sec_in_line:(line + 1) * self.fs * self.sec_in_line]
        step = self.px_in_1sec / self.fs
        line_array = [(j * step, self.px_in_1mv * 3 + self.px_in_1mv * 2 - round(line_data[j] * self.px_in_1mv))
                      for j in range(len(line_data))]
        self.canvas_left.create_line(line_array)
        self.canvas_right.create_line(line_array)

        r_peaks = self.r_peaks_left[
            np.where((self.r_peaks_left >= line * self.fs * self.sec_in_line) &
                     (self.r_peaks_left < (line + 1) * self.fs * self.sec_in_line))[0]]
        r_peaks = r_peaks - (line * self.fs * self.sec_in_line)

        for j in range(len(r_peaks)):
            self.canvas_left.create_oval(
                r_peaks[j] * step - 1, line_array[r_peaks[j]][1] - 1,
                r_peaks[j] * step + 1, line_array[r_peaks[j]][1] + 1,
                fill='red', outline='red')

    def draw_graph(self):
        total_lines = len(self.data) // (self.fs * self.sec_in_line)
        for i in range(self.lines_load, min(total_lines, self.lines_load + self.lines_add)):
            line_data = self.data[i * self.fs * self.sec_in_line:(i + 1) * self.fs * self.sec_in_line]
            filtered_line_data = self.filtered[i * self.fs * self.sec_in_line:(i + 1) * self.fs * self.sec_in_line]
            step = self.px_in_1sec / self.fs
            line_array, filtered_line_array = [], []
            for j in range(len(line_data)):
                line_array.append(
                    (j * step, i * self.px_in_1mv * 3 + self.px_in_1mv * 2 - round(line_data[j] * self.px_in_1mv)))
                filtered_line_array.append(
                    (j * step,
                     i * self.px_in_1mv * 3 + self.px_in_1mv * 2 - round(filtered_line_data[j] * self.px_in_1mv)))
            if self.vs_filtered:
                self.canvas_left.create_line(filtered_line_array)
                self.canvas_right.create_line(line_array)
            else:
                self.canvas_left.create_line(filtered_line_array)
                self.canvas_right.create_line(filtered_line_array)

            if len(self.r_peaks_left) > 0:
                # if self.vs_filtered:
                r_peak_range = np.where((np.array(self.r_peaks_left) >= i * self.fs * self.sec_in_line) &
                                        (np.array(self.r_peaks_left) < (i + 1) * self.fs * self.sec_in_line))[0]

                r_peaks = np.array(self.r_peaks_left)[r_peak_range]
                r_peaks = r_peaks - (i * self.fs * self.sec_in_line)

                for j in range(len(r_peaks)):
                    if self.qrs_type[r_peak_range[j]] == 1:
                        color = 'red'
                    elif self.qrs_type[r_peak_range[j]] == 2:
                        color = 'green'
                        self.canvas_left.create_rectangle(
                            r_peaks[j] * step - 2, filtered_line_array[r_peaks[j]][1] - 2,
                            r_peaks[j] * step + 2, filtered_line_array[r_peaks[j]][1] + 2,
                            fill=color, outline=color
                        )
                        continue
                    elif self.qrs_type[r_peak_range[j]] == 3:
                        color = 'blue'
                        self.canvas_left.create_rectangle(
                            r_peaks[j] * step - 2, filtered_line_array[r_peaks[j]][1] - 2,
                            r_peaks[j] * step + 2, filtered_line_array[r_peaks[j]][1] + 2,
                            fill=color, outline=color
                        )
                        continue
                    else:
                        color = 'red'

                    self.canvas_left.create_oval(
                        r_peaks[j] * step - 2, filtered_line_array[r_peaks[j]][1] - 2,
                        r_peaks[j] * step + 2, filtered_line_array[r_peaks[j]][1] + 2,
                        fill=color, outline=color)
                # else:
                #     r_peaks = self.r_peaks_left[
                #         np.where((self.r_peaks_left >= i * self.fs * self.sec_in_line) &
                #                  (self.r_peaks_left < (i + 1) * self.fs * self.sec_in_line))[0]]
                #     r_peaks = r_peaks - (i * self.fs * self.sec_in_line)
                #
                #     for j in range(len(r_peaks)):
                #         self.canvas_left.create_oval(
                #             r_peaks[j] * step - 2, line_array[r_peaks[j]][1] - 2,
                #             r_peaks[j] * step + 2, line_array[r_peaks[j]][1] + 2,
                #             fill='red', outline='red')

            if len(self.r_peaks_right) > 0:
                if self.vs_filtered:
                    ref_data = line_array
                else:
                    ref_data = filtered_line_array

                r_peak_range = np.where((np.array(self.r_peaks_right) >= i * self.fs * self.sec_in_line) &
                                        (np.array(self.r_peaks_right) < (i + 1) * self.fs * self.sec_in_line))[0]

                r_peaks = np.array(self.r_peaks_right)[r_peak_range]
                r_peaks = r_peaks - (i * self.fs * self.sec_in_line)

                for j in range(len(r_peaks)):
                    if self.qrs_type[r_peak_range[j]] == 1:
                        color = 'red'
                    elif self.qrs_type[r_peak_range[j]] == 2:
                        color = 'green'
                        self.canvas_right.create_rectangle(
                            r_peaks[j] * step - 2, ref_data[r_peaks[j]][1] - 2,
                            r_peaks[j] * step + 2, ref_data[r_peaks[j]][1] + 2,
                            fill=color, outline=color
                        )
                        continue
                    elif self.qrs_type[r_peak_range[j]] == 3:
                        color = 'blue'
                        self.canvas_right.create_rectangle(
                            r_peaks[j] * step - 2, ref_data[r_peaks[j]][1] - 2,
                            r_peaks[j] * step + 2, ref_data[r_peaks[j]][1] + 2,
                            fill=color, outline=color
                        )
                        continue
                    else:
                        color = 'red'

                    self.canvas_right.create_oval(
                        r_peaks[j] * step - 2, ref_data[r_peaks[j]][1] - 2,
                        r_peaks[j] * step + 2, ref_data[r_peaks[j]][1] + 2,
                        fill=color, outline=color)

            # self.canvas_info.create_text(20, i * self.px_in_1mv * 3 + self.px_in_1mv * 2, text=str(len(r_peaks)))
            self.canvas_line.create_text(20, i * self.px_in_1mv * 3 + self.px_in_1mv * 2, text=str(i))
            print('\r', i, len(self.data) // (self.fs * self.sec_in_line), end='', file=sys.stderr)
        print()
        self.lines_load += self.lines_add
        scroll_region = (0, 0, self.canvas_width, self.px_in_1mv * 3 * min(total_lines, self.lines_load))
        self.canvas_left.configure(scrollregion=scroll_region)
        self.canvas_right.configure(scrollregion=scroll_region)
        self.canvas_info.configure(scrollregion=scroll_region)
        self.canvas_line.configure(scrollregion=scroll_region)

    def detector_for_left(self):
        print('peak detection for left')
        start = time.perf_counter()
        ecg_clean = nk.ecg_clean(self.data, sampling_rate=self.fs, method='biosppy')
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs, method='neurokit')
        self.r_peaks_left = rpeaks['ECG_R_Peaks']
        print('elapsed:', time.perf_counter() - start)


if __name__ == '__main__':
    comparison = Comparison()
    comparison.make_gui()


