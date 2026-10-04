from PIL import Image

from paligemma2_experiment.images import degrade_then_restore


def test_degraded_image_has_target_dimensions():
    image = Image.new("RGB", (900, 600), color="red")
    degraded = degrade_then_restore(image, low_resolution=224, target_resolution=448)
    assert degraded.mode == "RGB"
    assert degraded.size == (448, 448)
