import torch


def initialize(model, path, expand_input=False):
    """Strict weights import with explicit 10 -> 14 expansion when requested."""

    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if "learner" in saved:
        weights = {
            **{
                "learner." + key: value
                for key, value in saved["learner"].items()
            },
            **{
                "last_layer_learner." + key: value
                for key, value in saved["last_layer_learner"].items()
            },
        }

        if weights["learner.conv_0.weight"].shape[1] != 10:
            raise ValueError(
                "Legacy Hermite checkpoint omits fixed filters; "
                "use the known 10-channel baseline or a full new checkpoint"
            )

        legacy = True
    else:
        weights = saved.get("model", saved)
        legacy = False

    current = model.state_dict()
    trainable = set(
        dict(model.named_parameters())
    )

    # Old state_dict omitted generated buffers; keep local known filters,
    # but require every trainable tensor and every BN statistic from that
    # checkpoint.
    required = (
        {
            key
            for key in current
            if key.startswith(
                (
                    "learner.",
                    "last_layer_learner.",
                )
            )
        }
        if legacy
        else set(current)
    )

    missing = required - set(weights)
    unexpected = set(weights) - set(current)

    if missing or unexpected:
        raise ValueError(
            "Checkpoint mismatch: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    expanded = False

    for key, value in weights.items():
        if key not in current:
            raise ValueError(
                f"Unexpected tensor {key}"
            )

        if value.shape != current[key].shape:
            can_expand_input = (
                key == "learner.conv_0.weight"
                and expand_input
                and value.shape[1] == 10
                and current[key].shape[1] == 14
                and value.shape[0] == current[key].shape[0]
                and value.shape[2:] == current[key].shape[2:]
            )

            if can_expand_input:
                target = torch.zeros_like(
                    current[key]
                )
                target[:, :10] = value
                weights[key] = target
                expanded = True
            else:
                raise ValueError(
                    f"{key}: checkpoint {tuple(value.shape)} "
                    f"vs model {tuple(current[key].shape)}. "
                    "No layer was skipped."
                )

    if expand_input and not expanded:
        raise ValueError(
            "--expand-input requested but no 10-to-14 layer was found"
        )

    current.update(weights)

    model.load_state_dict(
        current,
        strict=True,
    )

    return {
        "legacy": legacy,
        "expanded_input": expanded,
        "loaded_trainable_tensors": len(trainable),
    }