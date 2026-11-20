"""Ground-truth annotations are validated against their image."""

from __future__ import annotations

import pytest

from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.geometry import BBox


def test_annotation_keeps_class_and_box() -> None:
    annotation = Annotation(DamageClass.D00, BBox(1, 2, 30, 40))

    assert annotation.damage_class is DamageClass.D00
    assert annotation.bbox == BBox(1, 2, 30, 40)


def test_annotated_image_copies_annotations_into_a_tuple() -> None:
    annotation = Annotation(DamageClass.D20, BBox(0, 0, 5, 5))

    image = AnnotatedImage(
        image_id="Japan/000001.jpg",
        width=600,
        height=600,
        annotations=[annotation],  # type: ignore[arg-type]
    )

    assert image.image_id == "Japan/000001.jpg"
    assert image.annotations == (annotation,)
    assert image.width == 600


def test_annotated_image_allows_negative_samples() -> None:
    image = AnnotatedImage(image_id="Czech/000002.jpg", width=600, height=600)

    assert image.annotations == ()


@pytest.mark.parametrize(("width", "height"), [(0, 600), (600, 0), (-1, 600)])
def test_annotated_image_requires_positive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        AnnotatedImage(image_id="x.jpg", width=width, height=height)


@pytest.mark.parametrize("image_id", ["", "   "])
def test_annotated_image_requires_an_image_id(image_id: str) -> None:
    with pytest.raises(ValueError, match="image_id"):
        AnnotatedImage(image_id=image_id, width=10, height=10)


@pytest.mark.parametrize(
    "box",
    [
        BBox(-1, 0, 5, 5),
        BBox(0, 0, 11, 5),
        BBox(0, -2, 5, 5),
        BBox(0, 0, 5, 11),
    ],
)
def test_annotated_image_rejects_boxes_outside_the_image(box: BBox) -> None:
    with pytest.raises(ValueError, match="outside"):
        AnnotatedImage(
            image_id="x.jpg",
            width=10,
            height=10,
            annotations=(Annotation(DamageClass.D10, box),),
        )


def test_annotated_image_accepts_boxes_on_the_border() -> None:
    image = AnnotatedImage(
        image_id="x.jpg",
        width=10,
        height=10,
        annotations=(
            Annotation(DamageClass.D10, BBox(0, 0, 10, 10)),
            Annotation(DamageClass.D40, BBox(10, 10, 10, 10)),
        ),
    )

    assert len(image.annotations) == 2
