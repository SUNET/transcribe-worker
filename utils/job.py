from enum import Enum


class JobStatusEnum(str, Enum):
    """
    Enum representing the status of a job.
    """

    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
