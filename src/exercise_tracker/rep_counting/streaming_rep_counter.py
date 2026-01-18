from typing import Optional, List, Literal


class StreamingRepCounter:
    """
    To count reps from streamed phase series
    """

    phase_buffer: Optional[List[float]] = None
    stream_rep_phase: Literal["rep-started"] | Literal["rep-end"] = None
    num_reps: int = 0

    def __init__(
            self,
            min_phase_extrema_duration=0.1,
            max_rep_start_phase=0.2,
            min_rep_end_phase=0.75,
            min_phase_start_duration=None,
            min_phase_end_duration=None,
    ):
        self.min_phase_extrema_duration = min_phase_extrema_duration
        self.max_rep_start_phase = max_rep_start_phase
        self.min_rep_end_phase = min_rep_end_phase
        self.min_phase_start_duration = min_phase_start_duration or min_phase_start_duration
        self.min_phase_end_duration = min_phase_end_duration or min_phase_end_duration


    def ingest_phase(self, phase: float, return_reps=False) -> None | int:
        """
        If we've been under max_rep_start_phase for min_phase_start_duration - we start a rep
        and if we've been over min_rep_end_phase for min_phase_end_duration - we end a rep.

        We also consider the 'velocity' of the phase.

        """
        ...


        if return_reps:
            return self.get_number_of_reps()

    
    def get_number_of_reps(self) -> int:

        return 0