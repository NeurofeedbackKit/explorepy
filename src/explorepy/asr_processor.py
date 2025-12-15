import time
from copy import deepcopy

from eegprep import clean_asr
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AsrProcessor:
    _min_calibration_length: float = 10.  # in s
    _default_calibration_length: float = 30.  # in s
    _max_calibration_length: float = 120.  # in s

    _min_cutoff: float = 3.0
    _default_cutoff: float = 5.0
    _max_cutoff: float = 20.0

    _min_clean_timer: float = 0.5  # in s
    _default_clean_timer: float = 1.0  # in s
    _max_clean_timer: float = 3.0  # in s

    def __init__(self, stream_proc, in_topic):
        self.stream_processor = stream_proc
        self.in_topic = in_topic
        self.is_initialized = False
        self.cleaned_data_available = False
        self.cleaned_data = {}
        info_keys = self.stream_processor.device_info.keys()
        if "sampling_rate" not in info_keys or "firmware_version" not in info_keys:
            logger.error("Sampling rate or firmware version not available from stream processor, "
                         "cannot instantiate AsrProcessor!")
            return
        self.sr = self.stream_processor.device_info["sampling_rate"]
        fw = self.stream_processor.device_info["firmware_version"]
        self.ch_count = 8 if fw[0] == '7' else 16 if fw[0] == '8' else 32

        self.calibration_data = np.empty(shape=(self.ch_count, 0))
        self.is_calibrating = False
        self.calibration_data_available = False

        self.asr_packet_count = 0
        self.calib_started_at: float = -1.0
        self.calibration_length: float = self._default_calibration_length  # in s

        self.last_clean_at: float = -1.0  # the last time the to_clean buffer was cleaned
        self.clean_timer: float = self._default_clean_timer  # how long to wait between running asr, in s

        self.cutoff = self._default_cutoff

        self.to_clean = np.empty(shape=(self.ch_count, 0))
        self.to_clean_ts = np.empty(shape=0)
        self.is_initialized = True

    def update_device_data(self, ch_count: int, sr: float):
        self.ch_count = ch_count
        self.sr = sr

        self.calibration_data = np.empty(shape=(self.ch_count, 0))
        self.to_clean = np.empty(shape=(self.ch_count, 0))
        self.to_clean_ts = np.empty(shape=(1, 0))

    def clear_calibration_data(self):
        self.calibration_data = np.empty(shape=(self.ch_count, 0))

    def clear_data_buffer(self):
        self.to_clean = np.empty(shape=(self.ch_count, 0))
        self.to_clean_ts = np.empty(shape=(1, 0))

    def on_calib_data_received(self, packet):
        if (self.calib_started_at <= 0.0 or
            not self._min_calibration_length <= self.calibration_length <= self._max_calibration_length):
            raise ValueError(
                "Error writing calibration packet, timer has not been set correctly or calibration length is invalid!")
        if (time.time() - self.calib_started_at) > self.calibration_length:
            self.calibration_data_available = True
            self.stop_calibration()
            return
        self.calibration_data = np.append(self.calibration_data, packet.get_data()[1], axis=1)  # TODO np array, check shape

    def on_unclean_data_received(self, packet):
        if self.last_clean_at <= 0.0:
            self.last_clean_at = time.time()
        if not self.calibration_data_available:
            logger.warning("Attempting to clean data with no calibration available - returning...")
        self.to_clean = np.append(self.to_clean, packet.get_data()[1], axis=1)
        self.to_clean_ts = np.append(self.to_clean_ts, packet.get_data()[0])
        if time.time() - self.last_clean_at >= self.clean_timer:
            self.clean_data()
            self.clear_data_buffer()
            self.last_clean_at = -1.0

    def clear_cleaned_data(self):
        self.cleaned_data_available = False
        self.cleaned_data = {}

    def clean_data(self):
        asr_dict = {"data": self.to_clean.copy(),
                    "srate": self.sr,
                    "nbchan": self.ch_count,
                    "timestamps": self.to_clean_ts.copy()}
        try:
            ret = clean_asr(deepcopy(asr_dict), cutoff=self.cutoff, window_len=None, ref_maxbadchannels=self.calibration_data)
            self.cleaned_data_available = True
            self.cleaned_data = ret
        except ValueError:
            logger.error("Could not get ASR from input window!")

    def start_cleaning(self, clean_timer: float = None):
        if clean_timer is None:
            self.clean_timer = self._default_clean_timer
        elif self._min_clean_timer <= clean_timer <= self._max_clean_timer:
            self.clean_timer = clean_timer
        else:
            logger.error(f"Passed refresh timer for ASR of {clean_timer} is not within accepted range of "
                         f"[{self._min_clean_timer},{self._max_clean_timer}]")
        logger.info(f"Starting cleaning with ASR (refresh every {self.clean_timer}s)...")
        self.stream_processor.subscribe(self.on_unclean_data_received, topic=self.in_topic)

    def stop_cleaning(self):
        logger.info("Stopping cleaning with ASR.")
        self.stream_processor.unsubscribe(self.on_unclean_data_received, topic=self.in_topic)

    def start_calibration(self, calib_length: float=-1.0):
        self.is_calibrating = True
        if self._min_calibration_length <= calib_length <= self._max_calibration_length:
            self.calibration_length = calib_length
        else:
            logger.error(f"Passed refresh timer for ASR of {calib_length} is not within accepted range of "
                         f"[{self._min_calibration_length},{self._max_calibration_length}]")
        logger.info(f"Starting ASR calibration for {self.calibration_length}s...")
        self.calib_started_at = time.time()
        self.stream_processor.subscribe(self.on_calib_data_received, topic=self.in_topic)

    def stop_calibration(self):
        logger.info(f"Stopping ASR calibration.")
        self.stream_processor.unsubscribe(self.on_calib_data_received, topic=self.in_topic)
        self.is_calibrating = False
        self.calib_started_at = -1.0
        self.calibration_length = self._default_calibration_length

