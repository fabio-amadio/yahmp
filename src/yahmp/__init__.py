from mjlab.utils.lab_api.tasks.importer import import_packages

from yahmp.utils import get_wandb_checkpoint_path as _get_wandb_checkpoint_path

_BLACKLIST_PKGS = ["scripts", ".mdp"]


def _patch_mjlab_wandb_checkpoint_loading() -> None:
  """Use YAHMP's W&B checkpoint resolver for mjlab train resume paths.

  The upstream mjlab helper only recognizes numbered checkpoints like
  ``model_1000.pt``. YAHMP runs often upload only ``model_latest.pt`` via the
  rolling-latest mode, so patch the helper used by ``mjlab.scripts.train`` to
  support both forms.
  """
  try:
    import mjlab.utils.os as mjlab_os

    mjlab_os.get_wandb_checkpoint_path = _get_wandb_checkpoint_path
  except Exception:
    pass

  try:
    import mjlab.scripts.train as mjlab_train

    mjlab_train.get_wandb_checkpoint_path = _get_wandb_checkpoint_path
  except Exception:
    pass


_patch_mjlab_wandb_checkpoint_loading()

import_packages(__name__, _BLACKLIST_PKGS)
