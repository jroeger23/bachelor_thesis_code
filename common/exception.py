import typing as t


class MisconfigurationError(Exception):
  """A Misconfiguration Exception type
  """

  def __init__(self, msg: t.Optional[str] = None) -> None:
    """Set Misconfiguration Error

    Args:
        msg (t.Optional[str], optional): The error message. Defaults to None.
    """
    self.msg = msg

  def __str__(self) -> str:
    return f'MisconfigurationError({self.msg})'