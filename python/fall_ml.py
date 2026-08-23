"""
UNO Q dual-IMU fall inference engine.

STAGE 3D.6 CANONICAL DEPLOYMENT PATH
-------------------------------------

W2:
    live waist m/s²
        -> [-X, -Y, +Z]
        -> / 9.80665
        -> frozen W2 normalization
        -> W2 ONNX

T1:
    live thigh values
        -> NO axis transformation
        -> NO unit conversion
        -> ACTIVE T1 channel-wise normalization
        -> T1 ONNX
        -> sigmoid(raw T1 logit)
        -> NO Platt calibration

Fusion:
    85% W2 probability
    15% T1 probability

Temporal:
    EMA alpha = 0.55
    threshold = 0.52

IMPORTANT:
The T1 normalization below is the active Stage 3D.3 /
Stage 3D.5 training-reference contract.

T1 channel order:
    [thigh_ax, thigh_ay, thigh_az]

T1 normalization:

    ax' = (ax - (-6.501838684082031)) / 3.9341483116149902
    ay' = (ay - (-1.0582934617996216)) / 10.274575233459473
    az' = (az - (-1.4698399305343628)) / 6.669708251953125
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import numpy as np
import onnxruntime as ort


# ============================================================
# DEPLOYMENT CONSTANTS
# ============================================================

WINDOW_SAMPLES = 250
STRIDE_SAMPLES = 125

G_TO_MS2 = 9.80665


# ============================================================
# W2 FROZEN NORMALIZATION
# ============================================================

WAIST_MEAN = np.asarray(
    [
        0.043933141976594925,
        -0.4208561182022095,
        0.0652138888835907,
    ],
    dtype=np.float32,
)

WAIST_STD = np.asarray(
    [
        0.17782951891422272,
        0.723371684551239,
        0.5445636510848999,
    ],
    dtype=np.float32,
)


# ============================================================
# ACTIVE T1 NORMALIZATION
#
# Source:
# T1_THIGH_A_2.5s active checkpoint contract
#
# Channel order:
# [thigh_ax, thigh_ay, thigh_az]
# ============================================================

THIGH_MEAN = np.asarray(
    [
        -6.501838684082031,
        -1.0582934617996216,
        -1.4698399305343628,
    ],
    dtype=np.float32,
)

THIGH_STD = np.asarray(
    [
        3.9341483116149902,
        10.274575233459473,
        6.669708251953125,
    ],
    dtype=np.float32,
)


# ============================================================
# ORIGINAL T1 PLATT CALIBRATION
#
# Retained only as historical reference.
# NOT USED.
# ============================================================

T1_CALIBRATOR_A = 10.866962009627777
T1_CALIBRATOR_B = -1.9291989676100594


# ============================================================
# FUSION / TEMPORAL CONTRACT
# ============================================================

WAIST_WEIGHT = 0.85
THIGH_WEIGHT = 0.15

EMA_ALPHA = 0.55
DECISION_THRESHOLD = 0.52


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid.
    """

    if x >= 0.0:

        z = np.exp(-x)

        return float(
            1.0 / (1.0 + z)
        )

    z = np.exp(x)

    return float(
        z / (1.0 + z)
    )


# ============================================================
# RESULT
# ============================================================

@dataclass
class FallInferenceResult:

    window_index: int

    w2_logit: float
    w2_probability: float

    t1_raw_logit: float
    t1_raw_probability: float

    # Kept for logger compatibility.
    #
    # IMPORTANT:
    # This is now deliberately identical to the raw
    # T1 probability because Platt calibration is disabled.
    t1_calibrated_probability: float

    fusion_probability: float
    ema_probability: float

    fall_decision: bool

    inference_latency_ms: float


# ============================================================
# DETECTOR
# ============================================================

class DualIMUFallDetector:
    """
    Streaming W2 + T1 inference.

    Input sample:

        waist_ax
        waist_ay
        waist_az
        thigh_ax
        thigh_ay
        thigh_az

    W2:

        raw waist m/s²
            -> [-X, -Y, +Z]
            -> / 9.80665
            -> W2 normalization
            -> ONNX

    T1:

        raw thigh
            -> NO axis transformation
            -> NO unit conversion
            -> ACTIVE T1 normalization
            -> ONNX

    Window:

        250 samples

    Stride:

        125 samples
    """

    def __init__(
        self,
        waist_model_path: str | Path,
        thigh_model_path: str | Path,
        *,
        verbose: bool = False,
    ) -> None:

        self.waist_model_path = Path(
            waist_model_path
        )

        self.thigh_model_path = Path(
            thigh_model_path
        )

        self.verbose = verbose

        # ----------------------------------------------------
        # Validate models
        # ----------------------------------------------------

        if not self.waist_model_path.exists():

            raise FileNotFoundError(
                self.waist_model_path
            )

        if not self.thigh_model_path.exists():

            raise FileNotFoundError(
                self.thigh_model_path
            )

        # ----------------------------------------------------
        # ONNX Runtime
        # ----------------------------------------------------

        self.waist_session = (
            ort.InferenceSession(
                str(self.waist_model_path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.thigh_session = (
            ort.InferenceSession(
                str(self.thigh_model_path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.waist_input_name = (
            self.waist_session
            .get_inputs()[0]
            .name
        )

        self.thigh_input_name = (
            self.thigh_session
            .get_inputs()[0]
            .name
        )

        # ----------------------------------------------------
        # Rolling buffers
        # ----------------------------------------------------

        self.waist_buffer: deque[
            tuple[float, float, float]
        ] = deque(
            maxlen=WINDOW_SAMPLES
        )

        self.thigh_buffer: deque[
            tuple[float, float, float]
        ] = deque(
            maxlen=WINDOW_SAMPLES
        )

        # ----------------------------------------------------
        # Streaming state
        # ----------------------------------------------------

        self.samples_since_inference = 0

        self.window_index = 0

        self.ema_probability: Optional[
            float
        ] = None

        # ----------------------------------------------------
        # Deployment contract announcement
        # ----------------------------------------------------

        if self.verbose:

            print()
            print("=" * 70)
            print("DUAL IMU FALL DETECTOR")
            print("STAGE 3D.6 CANONICAL T1 PATH")
            print("=" * 70)

            print(
                "W2 model:",
                self.waist_model_path,
            )

            print(
                "T1 model:",
                self.thigh_model_path,
            )

            print()
            print("W2 transformation:")
            print(
                "  raw m/s²"
                " -> [-X,-Y,+Z]"
                " -> /9.80665"
                " -> W2 normalization"
            )

            print()
            print("T1 transformation:")
            print(
                "  raw thigh"
                " -> NO axis transform"
                " -> NO unit conversion"
            )

            print()
            print("T1 normalization:")

            print(
                "  mean:",
                THIGH_MEAN.tolist(),
            )

            print(
                "  std :",
                THIGH_STD.tolist(),
            )

            print()
            print("T1 calibration:")
            print("  DISABLED")

            print()
            print("Fusion:")
            print(
                f"  {WAIST_WEIGHT:.2f} * W2"
                f" + {THIGH_WEIGHT:.2f} * T1"
            )

            print()
            print("EMA alpha:")
            print(f"  {EMA_ALPHA}")

            print()
            print("Decision threshold:")
            print(f"  {DECISION_THRESHOLD}")

            print("=" * 70)

    # ========================================================
    # W2 PREPROCESSING
    # ========================================================

    @staticmethod
    def _prepare_waist_window(
        waist_window: np.ndarray,
    ) -> np.ndarray:
        """
        W2 canonical preprocessing.

        Input:
            [250, 3]

        Units:
            m/s²

        Axis transform:
            X -> -X
            Y -> -Y
            Z -> +Z

        Unit conversion:
            m/s² -> g

        Normalization:
            frozen W2 mean/std

        Output:
            [1, 3, 250]
        """

        raw = np.asarray(
            waist_window,
            dtype=np.float32,
        )

        if raw.shape != (
            WINDOW_SAMPLES,
            3,
        ):

            raise ValueError(
                "Expected waist window "
                f"{(WINDOW_SAMPLES, 3)}, "
                f"got {raw.shape}"
            )

        # ----------------------------------------------------
        # LOCAL SENSOR -> CANONICAL WAIST
        # ----------------------------------------------------

        canonical = raw.copy()

        canonical[:, 0] *= -1.0
        canonical[:, 1] *= -1.0

        # Z remains unchanged.

        # ----------------------------------------------------
        # m/s² -> g
        # ----------------------------------------------------

        canonical /= np.float32(
            G_TO_MS2
        )

        # ----------------------------------------------------
        # FROZEN W2 NORMALIZATION
        # ----------------------------------------------------

        normalized = (
            canonical
            - WAIST_MEAN.reshape(1, 3)
        ) / WAIST_STD.reshape(1, 3)

        # ----------------------------------------------------
        # [250,3] -> [1,3,250]
        # ----------------------------------------------------

        model_input = np.transpose(
            normalized,
            (1, 0),
        )

        model_input = np.expand_dims(
            model_input,
            axis=0,
        )

        return np.ascontiguousarray(
            model_input,
            dtype=np.float32,
        )

    # ========================================================
    # T1 CANONICAL PREPROCESSING
    # ========================================================

    @staticmethod
    def _prepare_thigh_window(
        thigh_window: np.ndarray,
    ) -> np.ndarray:
        """
        ACTIVE T1 CANONICAL PREPROCESSING.

        Input:
            [250,3]

        Channel order:
            thigh_ax
            thigh_ay
            thigh_az

        Transformations:

            NO axis transformation
            NO sign flip
            NO rotation
            NO unit conversion
            NO filtering
            NO clipping

        Normalization:

            (x - THIGH_MEAN) / THIGH_STD

        Output:
            [1,3,250]
        """

        raw = np.asarray(
            thigh_window,
            dtype=np.float32,
        )

        if raw.shape != (
            WINDOW_SAMPLES,
            3,
        ):

            raise ValueError(
                "Expected thigh window "
                f"{(WINDOW_SAMPLES, 3)}, "
                f"got {raw.shape}"
            )

        # ----------------------------------------------------
        # ACTIVE T1 NORMALIZATION
        #
        # [250,3]
        # ----------------------------------------------------

        normalized = (
            raw
            - THIGH_MEAN.reshape(1, 3)
        ) / THIGH_STD.reshape(1, 3)

        # ----------------------------------------------------
        # [250,3] -> [1,3,250]
        # ----------------------------------------------------

        model_input = np.transpose(
            normalized,
            (1, 0),
        )

        model_input = np.expand_dims(
            model_input,
            axis=0,
        )

        return np.ascontiguousarray(
            model_input,
            dtype=np.float32,
        )

    # ========================================================
    # ONNX INFERENCE
    # ========================================================

    @staticmethod
    def _run_logit(
        session: ort.InferenceSession,
        input_name: str,
        model_input: np.ndarray,
    ) -> float:

        output = session.run(
            None,
            {
                input_name: model_input,
            },
        )[0]

        return float(
            np.asarray(
                output
            ).reshape(-1)[0]
        )

    # ========================================================
    # WINDOW INFERENCE
    # ========================================================

    def infer_window(
        self,
        waist_window: np.ndarray,
        thigh_window: np.ndarray,
    ) -> FallInferenceResult:

        waist_window = np.asarray(
            waist_window,
            dtype=np.float32,
        )

        thigh_window = np.asarray(
            thigh_window,
            dtype=np.float32,
        )

        if waist_window.shape != (
            WINDOW_SAMPLES,
            3,
        ):

            raise ValueError(
                f"waist_window must be "
                f"{(WINDOW_SAMPLES, 3)}, "
                f"got {waist_window.shape}"
            )

        if thigh_window.shape != (
            WINDOW_SAMPLES,
            3,
        ):

            raise ValueError(
                f"thigh_window must be "
                f"{(WINDOW_SAMPLES, 3)}, "
                f"got {thigh_window.shape}"
            )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        waist_input = (
            self._prepare_waist_window(
                waist_window
            )
        )

        thigh_input = (
            self._prepare_thigh_window(
                thigh_window
            )
        )

        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        start = time.perf_counter()

        w2_logit = self._run_logit(
            self.waist_session,
            self.waist_input_name,
            waist_input,
        )

        t1_raw_logit = self._run_logit(
            self.thigh_session,
            self.thigh_input_name,
            thigh_input,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        w2_probability = sigmoid(
            w2_logit
        )

        t1_raw_probability = sigmoid(
            t1_raw_logit
        )

        # ----------------------------------------------------
        # NO PLATT CALIBRATION
        #
        # Field retained for compatibility with
        # existing logger/main.py.
        # ----------------------------------------------------

        t1_calibrated_probability = (
            t1_raw_probability
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion_probability = (
            WAIST_WEIGHT
            * w2_probability
            +
            THIGH_WEIGHT
            * t1_calibrated_probability
        )

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        if self.ema_probability is None:

            self.ema_probability = (
                fusion_probability
            )

        else:

            self.ema_probability = (
                EMA_ALPHA
                * fusion_probability
                +
                (
                    1.0 - EMA_ALPHA
                )
                * self.ema_probability
            )

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        fall_decision = (
            self.ema_probability
            >= DECISION_THRESHOLD
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = FallInferenceResult(

            window_index=self.window_index,

            w2_logit=w2_logit,

            w2_probability=(
                w2_probability
            ),

            t1_raw_logit=t1_raw_logit,

            t1_raw_probability=(
                t1_raw_probability
            ),

            t1_calibrated_probability=(
                t1_calibrated_probability
            ),

            fusion_probability=(
                fusion_probability
            ),

            ema_probability=(
                self.ema_probability
            ),

            fall_decision=(
                fall_decision
            ),

            inference_latency_ms=(
                elapsed_ms
            ),
        )

        self.window_index += 1

        return result

    # ========================================================
    # STREAMING PUSH
    # ========================================================

    def push(
        self,
        waist_ax: float,
        waist_ay: float,
        waist_az: float,
        thigh_ax: float,
        thigh_ay: float,
        thigh_az: float,
    ) -> Optional[
        FallInferenceResult
    ]:

        # ----------------------------------------------------
        # Add samples
        # ----------------------------------------------------

        self.waist_buffer.append(
            (
                float(waist_ax),
                float(waist_ay),
                float(waist_az),
            )
        )

        self.thigh_buffer.append(
            (
                float(thigh_ax),
                float(thigh_ay),
                float(thigh_az),
            )
        )

        # ----------------------------------------------------
        # Wait for complete first window
        # ----------------------------------------------------

        if len(
            self.waist_buffer
        ) < WINDOW_SAMPLES:

            return None

        # ----------------------------------------------------
        # FIRST WINDOW
        # ----------------------------------------------------

        if self.window_index == 0:

            waist_window = np.asarray(
                self.waist_buffer,
                dtype=np.float32,
            )

            thigh_window = np.asarray(
                self.thigh_buffer,
                dtype=np.float32,
            )

            result = self.infer_window(
                waist_window,
                thigh_window,
            )

            self.samples_since_inference = 0

            return result

        # ----------------------------------------------------
        # SUBSEQUENT WINDOWS
        #
        # 125-sample stride
        # ----------------------------------------------------

        self.samples_since_inference += 1

        if (
            self.samples_since_inference
            < STRIDE_SAMPLES
        ):

            return None

        self.samples_since_inference = 0

        waist_window = np.asarray(
            self.waist_buffer,
            dtype=np.float32,
        )

        thigh_window = np.asarray(
            self.thigh_buffer,
            dtype=np.float32,
        )

        return self.infer_window(
            waist_window,
            thigh_window,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset_temporal_state(
        self,
    ) -> None:

        self.waist_buffer.clear()

        self.thigh_buffer.clear()

        self.samples_since_inference = 0

        self.window_index = 0

        self.ema_probability = None